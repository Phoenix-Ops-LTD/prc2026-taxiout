# SPDX-License-Identifier: GPL-3.0-only
"""Portable fitted forecast and typed inference contract; no production integration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pydantic import BaseModel, ConfigDict, Field, field_validator

from forecast_features import CATEGORIES, SCHEMA_VERSION, categorical, schedule_features
from pipeline import ID, PHASE, TARGET, TIME, sha256, write_json


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["mean", "median", "airport_time", "catboost", "lightgbm", "hist", "airport_lightgbm"]
    iterations: int = Field(default=500, ge=1, le=5000)
    depth: int = Field(default=7, ge=2, le=10)
    learning_rate: float = Field(default=.05, gt=0, le=1)
    threads: int = Field(default=6, ge=1, le=16)
    seed: int = 20261009
    demand: bool = True


class FittedForecast:
    """Fit-only vocabulary; unseen categories use UNKNOWN and baseline hierarchy."""
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.vocabulary: dict[str, list[str]] = {}
        self.estimator: Any = None
        self.specialists: dict[str, Any] = {}
        self.baseline: dict[str, Any] = {}
        self.columns: list[str] = []

    def encode(self, x: pd.DataFrame) -> pd.DataFrame:
        data = x[self.columns].copy()
        for col in CATEGORIES:
            values = data[col].where(data[col].isin(self.vocabulary[col]), "UNKNOWN")
            if self.spec.kind in ["lightgbm", "airport_lightgbm"]:
                data[col] = pd.Categorical(values, categories=self.vocabulary[col])
            elif self.spec.kind == "hist":
                # Explicit deterministic hash bins avoid HGB's 255-category ceiling.
                import hashlib
                data[col] = values.map(lambda v: int.from_bytes(hashlib.sha256(str(v).encode()).digest()[:2], "big") % 128).astype(float)
            else:
                data[col] = values.astype(str)
        return data

    def fit(self, x: pd.DataFrame, y: pd.Series[Any]) -> FittedForecast:
        if not len(x) or len(x) != len(y) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Fit requires matching nonnegative finite targets")
        self.columns = [str(c) for c in x.columns if self.spec.demand or not str(c).startswith("scheduled_")]
        self.vocabulary = {c: sorted(set(x[c].astype(str)) | {"UNKNOWN"}) for c in CATEGORIES}
        values = x.copy()
        values[TARGET] = y.to_numpy()
        self.baseline = {"mean": float(y.mean()), "median": float(y.median()),
                         "airport": values.groupby("ADEP_mvt")[TARGET].mean().to_dict()}
        grouped = values.groupby(["ADEP_mvt", "hour_utc"])[TARGET].agg(["mean", "count"])
        self.baseline["airport_hour"] = {str(r["ADEP_mvt"]) + ":" + str(int(r["hour_utc"])): float(r["mean"])
                                         for r in grouped.reset_index().to_dict("records") if r["count"] >= 50}
        fit = self.encode(x)
        if self.spec.kind == "catboost":
            self.estimator = CatBoostRegressor(iterations=self.spec.iterations, depth=self.spec.depth,
                learning_rate=self.spec.learning_rate, loss_function="RMSE", random_seed=self.spec.seed,
                thread_count=self.spec.threads, allow_writing_files=False, verbose=False)
            self.estimator.fit(fit, y, cat_features=CATEGORIES)
        elif self.spec.kind in ["lightgbm", "airport_lightgbm"]:
            params: dict[str, Any] = dict(n_estimators=self.spec.iterations, max_depth=self.spec.depth, num_leaves=63,
                          learning_rate=self.spec.learning_rate, min_child_samples=100, reg_lambda=10,
                          colsample_bytree=.9, n_jobs=self.spec.threads, random_state=self.spec.seed,
                          verbosity=-1, deterministic=True, force_col_wise=True)
            self.estimator = LGBMRegressor(**params)
            self.estimator.fit(fit, y, categorical_feature=CATEGORIES)
            if self.spec.kind == "airport_lightgbm":
                for airport in sorted(x["ADEP_mvt"].unique()):
                    mask = x["ADEP_mvt"].eq(airport)
                    if int(mask.sum()) >= 10000:
                        local = LGBMRegressor(**params)
                        local.fit(fit.loc[mask], y.loc[mask], categorical_feature=CATEGORIES)
                        self.specialists[str(airport)] = local
        elif self.spec.kind == "hist":
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.estimator = HistGradientBoostingRegressor(max_iter=self.spec.iterations, max_leaf_nodes=31,
                max_depth=self.spec.depth, learning_rate=self.spec.learning_rate, l2_regularization=10,
                min_samples_leaf=100, random_state=self.spec.seed, early_stopping=False,
                categorical_features=[c in CATEGORIES for c in self.columns])
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=self.spec.threads):
                self.estimator.fit(fit, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray[Any, Any]:
        fit = self.encode(x)
        if self.spec.kind in ["mean", "median"]:
            return np.full(len(x), self.baseline[self.spec.kind], dtype=float)
        if self.spec.kind == "airport_time":
            key = x["ADEP_mvt"] + ":" + x["hour_utc"].astype(int).astype(str)
            return key.map(self.baseline["airport_hour"]).fillna(x["ADEP_mvt"].map(self.baseline["airport"])).fillna(self.baseline["mean"]).to_numpy(dtype=float)
        if self.spec.kind == "hist":
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=self.spec.threads):
                prediction = np.asarray(self.estimator.predict(fit), dtype=float)
        else:
            prediction = np.asarray(self.estimator.predict(fit), dtype=float)
        for airport, local in self.specialists.items():
            mask = x["ADEP_mvt"].eq(airport).to_numpy()
            if mask.any():
                # Fixed shrinkage; no airport weights chosen from validation.
                prediction[mask] = .5 * prediction[mask] + .5 * local.predict(fit.loc[mask])
        unknown = ~x["ADEP_mvt"].isin(self.vocabulary["ADEP_mvt"]).to_numpy()
        prediction[unknown] = self.baseline["mean"]
        return np.maximum(0, prediction)

    def save(self, root: Path) -> dict[str, Any]:
        root.mkdir()
        metadata = {"spec": self.spec.model_dump(), "vocabulary": self.vocabulary,
                    "baseline": self.baseline, "columns": self.columns, "schema_version": SCHEMA_VERSION,
                    "artifacts": {}}
        if self.spec.kind == "catboost":
            self.estimator.save_model(str(root / "model.cbm"))
        elif self.spec.kind in ["lightgbm", "airport_lightgbm"]:
            self.estimator.booster_.save_model(str(root / "model.txt"))
            for i, (airport, model) in enumerate(sorted(self.specialists.items())):
                model.booster_.save_model(str(root / f"airport-{i}.txt"))
            metadata["specialist_files"] = {a: f"airport-{i}.txt" for i, a in enumerate(sorted(self.specialists))}
        elif self.spec.kind == "hist":
            import joblib
            joblib.dump(self.estimator, root / "model.joblib")
        metadata["artifacts"] = {p.name: sha256(p) for p in root.iterdir() if p.is_file()}
        write_json(root / "metadata.json", metadata)
        return metadata

    @classmethod
    def load(cls, root: Path) -> FittedForecast:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Forecast feature schema mismatch")
        for name, digest in metadata["artifacts"].items():
            if Path(name).name != name or sha256(root / name) != digest:
                raise ValueError("Model artifact hash mismatch")
        obj = cls(ModelSpec.model_validate(metadata["spec"]))
        obj.vocabulary, obj.baseline, obj.columns = metadata["vocabulary"], metadata["baseline"], metadata["columns"]
        if obj.spec.kind == "catboost":
            obj.estimator = CatBoostRegressor()
            obj.estimator.load_model(str(root / "model.cbm"))
        elif obj.spec.kind in ["lightgbm", "airport_lightgbm"]:
            from lightgbm import Booster
            obj.estimator = Booster(model_file=str(root / "model.txt"))
            obj.specialists = {a: Booster(model_file=str(root / f)) for a, f in metadata.get("specialist_files", {}).items()}
        elif obj.spec.kind == "hist":
            import joblib
            obj.estimator = joblib.load(root / "model.joblib")
        return obj


class PredictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    airport: str
    destination: str = "UNKNOWN"
    runway: str | None = None
    stand: str | None = None
    aircraft: str = "UNKNOWN"
    flightNumber: str | None = None
    scheduledTimeUtc: str
    scheduleKnownAtUtc: str

    @field_validator("scheduledTimeUtc", "scheduleKnownAtUtc")
    @classmethod
    def utc_time(cls, value: str) -> str:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None or stamp.utcoffset() != pd.Timedelta(0):
            raise ValueError("Explicit UTC timestamp required")
        return value


class PredictOutput(BaseModel):
    predictionSeconds: float = Field(ge=0)
    confidence: float | None = None
    range: tuple[float, float]
    modelVersion: str
    reasonCodes: list[str]


class TaxiOut:
    """taxiOut.predict() research adapter. Restricted weights never authorize product use."""
    def __init__(self, run: Path) -> None:
        self.manifest = json.loads((run / "model-manifest.json").read_text(encoding="utf-8"))
        self.models = {name: FittedForecast.load(run / name) for name in self.manifest["weights"]}

    def predict(self, request: PredictInput, schedule: pd.DataFrame) -> PredictOutput:
        known = pd.Timestamp(request.scheduleKnownAtUtc)
        if known > pd.Timestamp(request.scheduledTimeUtc):
            raise ValueError("Schedule snapshot must precede scheduled departure")
        row = pd.DataFrame({ID: ["__query__"], PHASE: ["DEP"], TIME: [request.scheduledTimeUtc],
                            "ADEP_mvt": [request.airport], "ADES_mvt": [request.destination],
                            "AIRCRAFT_TYPE_mvt": [request.aircraft], "FLIGHT_mvt": [request.flightNumber]})
        x = schedule_features(row, schedule)
        prediction = float(sum(self.manifest["weights"][n] * m.predict(x)[0] for n, m in self.models.items()))
        reasons = ["SCHEDULE_FORECAST", "EMPIRICAL_INTERVAL_NOT_CALIBRATED_PROBABILITY"]
        if request.runway or request.stand:
            reasons.append("UNVERIFIED_RUNWAY_STAND_UNUSED")
        for col, label in [("ADEP_mvt", "AIRPORT"), ("AIRCRAFT_TYPE_mvt", "AIRCRAFT")]:
            if any(x[col].iloc[0] not in m.vocabulary[col] for m in self.models.values()):
                reasons.append("UNSEEN_" + label)
        if self.manifest["data_class"] == "competition":
            reasons.append("RESEARCH_ONLY_COMPETITION_WEIGHTS")
        radius = float(self.manifest["interval_radius_seconds"])
        return PredictOutput(predictionSeconds=prediction, range=(max(0, prediction - radius), prediction + radius),
                             modelVersion=self.manifest["model_version"], reasonCodes=reasons)
