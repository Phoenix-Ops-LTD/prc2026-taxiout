# SPDX-License-Identifier: GPL-3.0-only
"""Freeze restricted forward-only worker inputs from an audited feature cache."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from forecast_features import clean_departures
from forecast_gpu_search import candidates, overrides, validate_inputs
from pipeline import ID, PHASE, TARGET, TIME, sha256, write_json


def prepare(data_root: Path, feature_run: Path, reference_search: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Use a new prepared-input directory")
    source = Path(__file__).parent
    feature_path = feature_run / "features.parquet"
    reference = json.loads((reference_search / "plan.json").read_text(encoding="utf-8"))
    if sha256(feature_path) != reference["feature_sha256"]:
        raise ValueError("Feature cache differs from the registered CPU experiment")
    audit = json.loads((feature_run / "audit.json").read_text(encoding="utf-8"))
    paths = sorted(data_root.glob("training_*.parquet"))
    if len(paths) != 12 or any(sha256(p) != audit["training"][p.name]["sha256"] for p in paths):
        raise ValueError("Training inputs differ from the twelve audited objects")
    columns = [ID, PHASE, TARGET, TIME, "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt"]
    data = clean_departures(pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True), training=True)
    features = pd.read_parquet(feature_path)
    if len(data) != len(features) or not data.index.equals(features.index):
        raise ValueError("Audited feature cache and departure order differ")
    mask = data[TIME].lt(pd.Timestamp("2025-12-01T00:00:00Z"))
    data, features = data.loc[mask].reset_index(drop=True), features.loc[mask].reset_index(drop=True)
    validate_inputs(data, features)
    output.mkdir(parents=True)
    data.to_parquet(output / "departures.parquet", index=False)
    features.to_parquet(output / "features.parquet", index=False)
    names = ["forecast_gpu_search.py", "forecast_features.py", "forecast_model.py", "forecast_guard.py", "forecast_novel_guard.py", "pipeline.py"]
    for name in names:
        shutil.copyfile(source / name, output / name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip())
    plan = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection_months": [9, 10, 11], "execution_unit": "GPU",
        "models": {n: s.model_dump() for n, s in candidates().items()}, "training_overrides": {n: overrides(n) for n in candidates()},
        "source_hashes": {n: sha256(output / n) for n in names},
        "input_hashes": {n: sha256(output / n) for n in ["departures.parquet", "features.parquet"]},
        "base_feature_sha256": sha256(feature_path), "training_input_hashes": {p.name: sha256(p) for p in paths},
        "git_provenance": {"git_commit": commit, "git_dirty_at_training": dirty, "commit_provenance": "EXACT_HEAD_AT_GPU_PLAN_FREEZE; exact source hashes recorded"},
        "rows": len(data), "official_feedback_used": False, "december_inputs_included": False, "previous_diagnostics_observed": True,
        "model_selection_rule": "Complete identical purged forward folds; fixed equal blends; minimum 1s pooled RMSE gain and every-fold improvement over incumbent before promotion",
        "gpu_memory_policy": "35% of free GPU memory; require at least 2600 MiB free before each fit; no existing job or service mutations",
        "gpu_training_determinism": "Non-deterministic GPU training; exact saved-model CPU inference parity",
        "preparer_sha256": sha256(Path(__file__))}
    write_json(output / "plan.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "feature-run", "reference-search", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.data, args.feature_run, args.reference_search, args.output)
    print(json.dumps({"rows": result["rows"], "input_hashes": result["input_hashes"]}, indent=2))
