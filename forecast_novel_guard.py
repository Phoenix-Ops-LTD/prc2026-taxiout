# SPDX-License-Identifier: GPL-3.0-only
"""Novel-aircraft fallback; preserve observed missing-category predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast_guard import POLICY as LEGACY_POLICY, GuardedForecast
from forecast_model import FittedForecast, ModelSpec
from pipeline import write_json

POLICY = "novel-aircraft-observed-missing-preserved/1.0"


class NovelGuardedForecast(GuardedForecast):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.missing_aircraft_seen_in_fit = False

    def fit(self, x: pd.DataFrame, y: pd.Series[Any]) -> NovelGuardedForecast:
        super().fit(x, y)
        self.missing_aircraft_seen_in_fit = bool(x["AIRCRAFT_TYPE_mvt"].eq("UNKNOWN").any())
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray[Any, Any]:
        prediction = FittedForecast.predict(self, x)
        unseen = (~x["AIRCRAFT_TYPE_mvt"].isin(self.vocabulary["AIRCRAFT_TYPE_mvt"])
            | (x["AIRCRAFT_TYPE_mvt"].eq("UNKNOWN") & (not self.missing_aircraft_seen_in_fit))).to_numpy()
        key = x["ADEP_mvt"] + ":" + x["hour_utc"].astype(int).astype(str)
        fallback = key.map(self.baseline["airport_hour"]).fillna(x["ADEP_mvt"].map(self.baseline["airport"])).fillna(self.baseline["mean"]).to_numpy(dtype=float)
        prediction[unseen] = fallback[unseen]
        return prediction

    def save(self, root: Path) -> dict[str, Any]:
        metadata = FittedForecast.save(self, root)
        metadata.update(guard_policy=POLICY, missing_aircraft_seen_in_fit=self.missing_aircraft_seen_in_fit)
        write_json(root / "metadata.json", metadata)
        return metadata

    @classmethod
    def load(cls, root: Path) -> NovelGuardedForecast:
        model = super().load(root)
        if not isinstance(model, cls):
            raise TypeError("Unexpected native model type")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("guard_policy") != POLICY or not isinstance(metadata.get("missing_aircraft_seen_in_fit"), bool):
            raise ValueError("Novel guard provenance is missing")
        model.missing_aircraft_seen_in_fit = metadata["missing_aircraft_seen_in_fit"]
        return model


def native_name(name: str) -> str:
    return name.removesuffix("_novel_guarded") if name.endswith("_novel_guarded") else name.removesuffix("_guarded")


def build_model(name: str, spec: ModelSpec) -> FittedForecast:
    if name.endswith("_novel_guarded"):
        return NovelGuardedForecast(spec)
    return GuardedForecast(spec) if name.endswith("_guarded") else FittedForecast(spec)


def load_model(root: Path) -> FittedForecast:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    policy = metadata.get("guard_policy")
    if policy is None:
        return FittedForecast.load(root)
    if policy == POLICY:
        return NovelGuardedForecast.load(root)
    if policy == LEGACY_POLICY:
        return GuardedForecast.load(root)
    raise ValueError("Unsupported saved prediction policy")
