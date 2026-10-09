# SPDX-License-Identifier: GPL-3.0-only
"""Independent fixed LightGBM depth search; selection months only, no final labels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from forecast_features import clean_departures, metrics, purged_split
from forecast_model import FittedForecast, ModelSpec
from forecast_run import log
from pipeline import ID, PHASE, TARGET, TIME, sha256, write_json


def search(data_root: Path, feature_run: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Use a new search directory")
    specs = {"shallow_no_demand": ModelSpec(kind="lightgbm", depth=5, iterations=1000, learning_rate=.035, demand=False, threads=3),
             "shallow_demand": ModelSpec(kind="lightgbm", depth=5, iterations=1000, learning_rate=.035, demand=True, threads=3),
             "deep_no_demand": ModelSpec(kind="lightgbm", depth=9, iterations=1000, learning_rate=.035, demand=False, threads=3)}
    output.mkdir(parents=True)
    feature_path = feature_run / "features.parquet"
    source_root = Path(__file__).parent
    write_json(output / "plan.json", {"status": "FIXED_BEFORE_DECEMBER_RESULTS", "months": [9, 10, 11],
        "models": {n: s.model_dump() for n, s in specs.items()}, "feature_sha256": sha256(feature_path),
        "source_hashes": {n: sha256(source_root / n) for n in ["forecast_search.py", "forecast_model.py", "forecast_features.py", "pipeline.py"]},
        "rankings_used": False, "final_targets_used": False})
    audit = json.loads((feature_run / "audit.json").read_text(encoding="utf-8"))
    paths = sorted(data_root.glob("training_*.parquet"))
    for path in paths:
        if sha256(path) != audit["training"][path.name]["sha256"]:
            raise ValueError("Verified training data digest changed")
    columns = [ID, PHASE, TIME, TARGET, "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt"]
    data = clean_departures(pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True), training=True)
    x = pd.read_parquet(feature_path)
    if len(x) != len(data) or not x.index.equals(data.index):
        raise ValueError("Feature and departure order differs")
    ledger: list[dict[str, Any]] = []
    oof: list[pd.DataFrame] = []
    for month in [9, 10, 11]:
        fit, valid, purged = purged_split(data, month)
        result = data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
        for name, spec in specs.items():
            log(f"PARALLEL FIT {month:02} {name}")
            started = time.monotonic()
            model = FittedForecast(spec).fit(x.loc[fit], data.loc[fit, TARGET])
            result[name] = model.predict(x.loc[valid])
            report = metrics(result[TARGET], result[name], result["ADEP_mvt"])
            ledger.append({"experiment": f"forward-{month:02}-{name}", "model": "lightgbm", "parameters": spec.model_dump(),
                "training_rows": int(fit.sum()), "purged_flight_rows": purged, "features": model.columns,
                "training_seconds": time.monotonic() - started, "validation_month": month, **report})
            write_json(output / "experiment-ledger.json", ledger)
            result.to_parquet(output / f"selection-{month:02}.parquet", index=False)
            log(f"PARALLEL RESULT {month:02} {name} RMSE={report['rmse']:.6f}")
        oof.append(result)
    pooled = pd.concat(oof, ignore_index=True)
    pooled.to_parquet(output / "selection-oof.parquet", index=False)
    scores = {n: metrics(pooled[TARGET], pooled[n], pooled["ADEP_mvt"]) for n in specs}
    write_json(output / "report.json", {"status": "FORWARD_SELECTION_ONLY_NO_DECEMBER_READ", "scores": scores})
    log("PARALLEL SEARCH COMPLETE " + json.dumps({n: r["rmse"] for n, r in scores.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    search(args.data, args.feature_run, args.output)
