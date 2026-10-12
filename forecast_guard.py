# SPDX-License-Identifier: GPL-3.0-only
"""Fixed training-only airport/hour fallback for unseen or missing aircraft."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast_model import FittedForecast, ModelSpec
from pipeline import write_json

POLICY = "unseen-aircraft-airport-hour/1.0"


class GuardedForecast(FittedForecast):
    def predict(self, x: pd.DataFrame) -> np.ndarray[Any, Any]:
        prediction = super().predict(x)
        unseen = (~x["AIRCRAFT_TYPE_mvt"].isin(self.vocabulary["AIRCRAFT_TYPE_mvt"])
                  | x["AIRCRAFT_TYPE_mvt"].eq("UNKNOWN")).to_numpy()
        key = x["ADEP_mvt"] + ":" + x["hour_utc"].astype(int).astype(str)
        fallback = key.map(self.baseline["airport_hour"]).fillna(x["ADEP_mvt"].map(self.baseline["airport"])).fillna(self.baseline["mean"]).to_numpy(dtype=float)
        prediction[unseen] = fallback[unseen]
        return prediction

    def save(self, root: Path) -> dict[str, Any]:
        metadata = super().save(root)
        metadata["guard_policy"] = POLICY
        write_json(root / "metadata.json", metadata)
        return metadata


def build_model(name: str, spec: ModelSpec) -> FittedForecast:
    return GuardedForecast(spec) if name.endswith("_guarded") else FittedForecast(spec)


def load_model(root: Path) -> FittedForecast:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    policy = metadata.get("guard_policy")
    if policy is None:
        return FittedForecast.load(root)
    if policy != POLICY:
        raise ValueError("Unsupported prediction guard policy")
    return GuardedForecast.load(root)
