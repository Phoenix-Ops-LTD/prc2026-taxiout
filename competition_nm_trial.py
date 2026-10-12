# SPDX-License-Identifier: GPL-3.0-only
"""Verify an existing competition-only NM model before preparing a new trial."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

FROZEN = Path(__file__).parent / "reproducibility/long-following-13.0.0"
ID, PHASE, TARGET = "MVT_ID_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt"

DEADLINE = datetime(2026, 10, 14, 10, tzinfo=timezone.utc)


def masked_inputs(raw: pd.DataFrame) -> pd.DataFrame:
    safe = raw.copy()
    dep = safe[PHASE].eq("DEP")
    safe.loc[dep, TARGET] = np.nan
    safe.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    return safe


def matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pipeline import departures
    from traffic_features import add_nm_observed_features, add_planned_features, enriched_features
    data = departures(raw)
    safe = masked_inputs(raw)
    inputs = departures(safe)
    x = add_nm_observed_features(add_planned_features(enriched_features(inputs, safe), inputs), inputs)
    return data, x


def prediction(model: CatBoostRegressor, x: pd.DataFrame) -> np.ndarray[Any, Any]:
    from traffic_features import nm_schedule_fallback_baseline
    return np.asarray(np.maximum(0, np.asarray(model.predict(x, thread_count=2)) + nm_schedule_fallback_baseline(x).to_numpy()), dtype=float)


def run(data_root: Path, model_run: Path, july_reference: Path, output: Path) -> None:
    sys.path.insert(0, str(FROZEN))
    import pipeline as preserved
    if Path(preserved.__file__).resolve().parent != FROZEN.resolve():
        raise ValueError("Run the competition trial in a fresh standalone interpreter")
    from pipeline import sha256, validate_submission, write_json
    started = time.monotonic()
    if (DEADLINE - datetime.now(timezone.utc)).total_seconds() < 7200 or output.exists():
        raise ValueError("Deadline reserve exhausted or output is not fresh")
    frozen_hashes = json.loads((FROZEN / "source-hashes.json").read_text(encoding="utf-8"))
    if any(Path(n).name != n or sha256(FROZEN / n) != h for n, h in frozen_hashes.items()):
        raise ValueError("Preserved source differs")
    report = json.loads((model_run / "report.json").read_text(encoding="utf-8"))
    if not report["experiment"]["residual"] or not report["experiment"]["schedule_fallback"] or report["experiment"]["features"] != "nm":
        raise ValueError("Unexpected fitted-model policy")
    if sha256(model_run / "model.cbm") != report["model_sha256"]:
        raise ValueError("Full model hash differs")
    paths = sorted(data_root.glob("training_*.parquet"))
    if len(paths) != 12 or any(sha256(p) != report["input_hashes"][p.name] for p in paths):
        raise ValueError("Training input binding differs")
    output.mkdir(parents=True)
    write_json(output / "plan.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "prc2026-nm-residual-standalone/1.0.0", "model_sha256": report["model_sha256"],
        "report_sha256": sha256(model_run / "report.json"), "source_sha256": sha256(Path(__file__)),
        "preserved_source_hashes": frozen_hashes, "training_input_hashes": report["input_hashes"],
        "ranking_sha256": sha256(data_root / "ranking.parquet"), "template_sha256": sha256(data_root / "submitting.parquet"),
        "competition_only": True, "predeparture_forecast": False, "new_fit_count": 0,
        "fixed_policy": "Preserved single CatBoost NM residual model + original NM/schedule fallback; no new weights",
        "maximum_cycle_seconds": 3600, "deadline_utc": DEADLINE.isoformat(), "minimum_reserve_seconds": 7200})
    print("RECOMPUTE_ALL_2025_TARGET_MASKED_FEATURES", flush=True)
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    data, x = matrix(raw)
    cached = pd.read_parquet(data_root.parent / "nm-features-v1.parquet")
    if list(x.columns) != report["features"]:
        raise ValueError("Original feature schema differs")
    pd.testing.assert_frame_equal(x, cached, check_dtype=False)
    del raw, cached
    verification = []
    for reference in [model_run, july_reference]:
        stored = pd.read_parquet(reference / "validation.parquet")
        expected = data.set_index(ID).loc[stored[ID]].reset_index()[stored.columns.drop("prediction")]
        pd.testing.assert_frame_equal(expected, stored.drop(columns="prediction"), check_dtype=False)
        positions = pd.Series(np.arange(len(data)), index=data[ID]).loc[stored[ID]].to_numpy()
        model = CatBoostRegressor()
        model.load_model(str(reference / "validation-model.cbm"))
        actual = prediction(model, x.iloc[positions])
        if not np.array_equal(actual, stored["prediction"].to_numpy()):
            raise ValueError("Preserved validation predictions differ")
        verification.append({"reference": reference.name, "rows": len(stored), "native_model_sha256": sha256(reference / "validation-model.cbm"),
            "native_prediction_parity": "EXACT", "validation_artifact_sha256": sha256(reference / "validation.parquet")})
        print("NATIVE_VALIDATION_EXACT", reference.name, len(stored), flush=True)
    del data, x
    if time.monotonic() - started > 3600 or (DEADLINE - datetime.now(timezone.utc)).total_seconds() < 7200:
        raise ValueError("Cycle budget or deadline reserve exhausted")
    raw = pd.read_parquet(data_root / "ranking.parquet")
    mask = raw[PHASE].eq("DEP")
    if raw.loc[mask, [TARGET, "BLOCK_TIME_UTC_mvt"]].notna().any().any():
        raise ValueError("Hidden departure targets or block times are populated")
    data, x = matrix(raw)
    model = CatBoostRegressor()
    model.load_model(str(model_run / "model.cbm"))
    if list(model.feature_names_) != report["features"] or model.tree_count_ != report["selected_iterations"]:
        raise ValueError("Native fitted-model contract differs")
    values = prediction(model, x)
    template = pd.read_parquet(data_root / "submitting.parquet")
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values, index=data[ID])).astype(float)
    validate_submission(template, result)
    artifact = output / "ranking_submission.parquet"
    result.to_parquet(artifact, index=False)
    fresh = CatBoostRegressor()
    fresh.load_model(str(model_run / "model.cbm"))
    if not np.array_equal(values, prediction(fresh, x)) or not pd.read_parquet(artifact).equals(result):
        raise ValueError("All-row native/serialization parity failed")
    manifest = {"data_class": "competition", "data_permission_ref": "PRC2026 registered participant, challenge-only",
        "model_version": "prc2026-nm-residual-standalone/1.0.0", "submission_sha256": sha256(artifact),
        "ranking_sha256": sha256(data_root / "ranking.parquet"), "template_sha256": sha256(data_root / "submitting.parquet"),
        "native_model_sha256": report["model_sha256"], "rows": len(result), "features": report["features"],
        "preserved_validation_parity": verification, "training_source_snapshot": frozen_hashes,
        "new_source_sha256": sha256(Path(__file__)), "native_all_ranking_rows_exact": True,
        "competition_only": True, "official_score": None, "elapsed_seconds": time.monotonic() - started,
        "status": "COMPLETE_NATIVE_VERIFIED_OFFICIAL_TRIAL_READY"}
    write_json(artifact.with_suffix(".manifest.json"), manifest)
    print("NM_TRIAL_ALL_ROWS_READY", len(result), manifest["submission_sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "model-run", "july-reference", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.model_run, args.july_reference, args.output)
