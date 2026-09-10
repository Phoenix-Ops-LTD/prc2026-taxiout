# SPDX-License-Identifier: GPL-3.0-only
"""Fixed five-seed completed-arrival specialist for unmatched LIRF movements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, ConfigDict, Field

from arrival_features import completed_arrival_features
from carrier_contest import matrix as carrier_matrix, specialist_scope
from pipeline import ID, TARGET, sha256, write_json
from traffic_features import nm_schedule_fallback_baseline

SEEDS = (20260907, 20260908, 20260909, 20260910, 20260911)


class SpecialistConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: Literal["prc2026-arrival-specialist/1.0.0"] = "prc2026-arrival-specialist/1.0.0"
    iterations: int = Field(default=1014, ge=1)
    data_permission_ref: str = Field(min_length=1, pattern=r"\S")


def matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dep, x = carrier_matrix(raw)
    x = x.loc[specialist_scope(x)].copy()
    dep = dep.loc[x.index]
    return dep, pd.concat([x, completed_arrival_features(dep, raw)], axis=1)


def train(data: Path, output: Path, config: SpecialistConfig) -> None:
    if output.exists():
        raise ValueError("Use a fresh immutable run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    hashes = {p.name: sha256(p) for p in paths}
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    clock = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
    if clock.isna().any() or not clock.dt.year.eq(2025).all():
        raise ValueError("Only complete 2025 movement clocks may enter training")
    dep, x = matrix(raw)
    del raw
    y = dep[TARGET].astype(float)
    if x.empty or not np.isfinite(y).all():
        raise ValueError("Specialist rows and finite labels are required")
    fit = y.ge(0)
    offset = nm_schedule_fallback_baseline(x)
    categories = list(x.select_dtypes(include=["object", "str"]).columns)
    output.mkdir(parents=True)
    write_json(output / "inputs.json", {"config": config.model_dump(), "input_hashes": hashes,
        "seeds": SEEDS, "features": list(x), "fit_rows": int(fit.sum())})
    experts: list[dict[str, Any]] = []
    for seed in SEEDS:
        print("FITTING_ARRIVAL_SPECIALIST", seed, int(fit.sum()), flush=True)
        model = CatBoostRegressor(iterations=config.iterations, depth=5, learning_rate=.04,
            l2_leaf_reg=10, loss_function="RMSE", random_seed=seed, thread_count=1,
            allow_writing_files=False, verbose=False)
        model.fit(x.loc[fit], (y - offset).loc[fit], cat_features=categories)
        path = output / f"seed-{seed}.cbm"
        model.save_model(str(path))
        experts.append({"seed": seed, "file": path.name, "sha256": sha256(path)})
        print("ARRIVAL_SPECIALIST_READY", seed, flush=True)
    write_json(output / "report.json", {"status": "FULL_FIT_NOT_OFFICIAL_SCORE",
        "config": config.model_dump(), "input_hashes": hashes, "features": list(x),
        "fit_rows": int(fit.sum()), "experts": experts, "equal_weight": .2})


def predict(run: Path, raw: pd.DataFrame) -> pd.Series:
    report = json.loads((run / "report.json").read_text())
    if report.get("status") != "FULL_FIT_NOT_OFFICIAL_SCORE":
        raise ValueError("Ranking prediction requires completed full-data experts")
    SpecialistConfig.model_validate(report["config"])
    experts = report["experts"]
    if [item["seed"] for item in experts] != list(SEEDS):
        raise ValueError("All five predefined expert seeds are required")
    dep, x = matrix(raw)
    if report["features"] != list(x):
        raise ValueError("Arrival specialist feature schema mismatch")
    offset = nm_schedule_fallback_baseline(x)
    predictions: list[np.ndarray] = []
    for item in experts:
        name = f"seed-{item['seed']}.cbm"
        if item["file"] != name or sha256(run / name) != item["sha256"]:
            raise ValueError("Arrival expert filename or digest mismatch")
        model = CatBoostRegressor()
        model.load_model(str(run / name))
        if list(model.feature_names_) != list(x):
            raise ValueError("Stored arrival expert schema mismatch")
        values = np.maximum(0, model.predict(x, thread_count=1) + offset) if len(x) else np.array([])
        predictions.append(np.asarray(values))
    return pd.Series(np.mean(predictions, axis=0), index=dep[ID], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    train(args.data, args.output, SpecialistConfig(data_permission_ref=args.permission_ref))


if __name__ == "__main__":
    main()
