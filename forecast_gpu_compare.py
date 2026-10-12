# SPDX-License-Identifier: GPL-3.0-only
"""Compare fixed equal pairs of frozen GPU candidates and incumbent components."""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast_gpu_search import candidates
from pipeline import ID, TARGET, TIME, sha256, write_json


def evaluate(frame: pd.DataFrame, incumbent_weights: dict[str, float], gpu_names: list[str]) -> dict[str, Any]:
    months = pd.to_datetime(frame[TIME], utc=True).dt.month
    if set(months) != {9, 10, 11} or not pd.to_datetime(frame[TIME], utc=True).dt.year.eq(2025).all() or frame[ID].duplicated().any():
        raise ValueError("Comparison requires unique complete 2025 forward fold rows")
    if not np.isclose(sum(incumbent_weights.values()), 1.) or any(w <= 0 for w in incumbent_weights.values()):
        raise ValueError("Invalid frozen incumbent weights")
    names = list(incumbent_weights) + gpu_names
    if len(set(names)) != len(names) or not np.isfinite(frame[names + [TARGET]]).all().all():
        raise ValueError("Missing, duplicate or nonfinite candidate predictions")

    def score(weights: dict[str, float]) -> dict[str, Any]:
        prediction = sum(w * frame[n].to_numpy() for n, w in weights.items())
        error = prediction - frame[TARGET].to_numpy()
        return {"weights": weights, "rmse": float(np.sqrt(np.mean(error ** 2))),
            "per_month": {str(m): float(np.sqrt(np.mean(error[months.eq(m).to_numpy()] ** 2))) for m in [9, 10, 11]}}

    incumbent = score(incumbent_weights)
    weights = {n: {n: 1.} for n in gpu_names}
    weights.update({a + "+" + b: {a: .5, b: .5} for a, b in itertools.combinations(names, 2) if a in gpu_names or b in gpu_names})
    scores = {n: score(w) for n, w in weights.items()}
    for result in scores.values():
        result["gain_vs_incumbent"] = incumbent["rmse"] - result["rmse"]
        result["eligible"] = result["gain_vs_incumbent"] >= 1. and all(result["per_month"][str(m)] < incumbent["per_month"][str(m)] for m in [9, 10, 11])
    eligible = [n for n, s in scores.items() if s["eligible"]]
    selected = min(eligible, key=lambda n: (scores[n]["rmse"], n)) if eligible else "+".join(incumbent_weights)
    return {"selected": selected, "promoted": bool(eligible), "incumbent": incumbent, "scores": scores,
        "leaderboard_used": False, "december_used_for_selection": False,
        "rule": "GPU singles and all fixed equal pairs among GPU candidates and frozen incumbent components; promotion requires >=1s pooled RMSE gain and improvement on every fold"}


def compare(gpu: Path, incumbent: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Use a new comparison directory")
    report = json.loads((gpu / "report.json").read_text(encoding="utf-8"))
    if not report["status"].startswith("COMPLETED_FORWARD_GPU_SEARCH"):
        raise ValueError("GPU batch is incomplete")
    a = pd.read_parquet(incumbent / "selection-oof.parquet")
    b = pd.read_parquet(gpu / "selection-oof.parquet")
    keys = [ID, TARGET, TIME, "ADEP_mvt"]
    if not a[keys].equals(b[keys]):
        raise ValueError("Incumbent and GPU forward identities, labels or order differ")
    selection = json.loads((incumbent / "selection.json").read_text(encoding="utf-8"))
    incumbent_weights = selection["scores"][selection["selected"]]["weights"]
    frame = a[keys + list(incumbent_weights)].copy()
    for name in candidates():
        frame[name] = b[name]
    result = evaluate(frame, incumbent_weights, list(candidates()))
    if not np.isclose(result["incumbent"]["rmse"], selection["scores"][selection["selected"]]["rmse"], rtol=0, atol=1e-8):
        raise ValueError("Frozen incumbent score changed")
    output.mkdir(parents=True)
    write_json(output / "plan.json", {"selection_months": [9, 10, 11], "fit_count": 0,
        "rule": result["rule"], "source_sha256": sha256(Path(__file__)),
        "input_hashes": {"gpu_oof": sha256(gpu / "selection-oof.parquet"), "incumbent_oof": sha256(incumbent / "selection-oof.parquet"),
                         "gpu_plan": sha256(gpu / "plan.json"), "incumbent_selection": sha256(incumbent / "selection.json")}})
    frame.to_parquet(output / "selection-oof.parquet", index=False)
    result["selected_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "selection.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["gpu", "incumbent", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.gpu, args.incumbent, args.output)
    print(json.dumps({"selected": result["selected"], "promoted": result["promoted"], "scores": {n: s["rmse"] for n, s in result["scores"].items()}}, indent=2))
