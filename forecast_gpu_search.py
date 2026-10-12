# SPDX-License-Identifier: GPL-3.0-only
"""Isolated bounded GPU search on the existing forecast allowlist and forward folds."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from forecast_features import CATEGORIES, metrics, purged_split
from forecast_model import ModelSpec
from forecast_novel_guard import NovelGuardedForecast, load_model
from pipeline import ID, TARGET, TIME, sha256, write_json


def candidates() -> dict[str, ModelSpec]:
    return {
        "gpu_deep_no_demand_novel_guarded": ModelSpec(kind="catboost", depth=8, iterations=3000, learning_rate=.035, demand=False, threads=2),
        "gpu_deep_demand_novel_guarded": ModelSpec(kind="catboost", depth=8, iterations=3000, learning_rate=.035, demand=True, threads=2),
        "gpu_smooth_no_demand_novel_guarded": ModelSpec(kind="catboost", depth=6, iterations=5000, learning_rate=.025, demand=False, threads=2),
    }


def overrides(name: str) -> dict[str, Any]:
    return {"task_type": "GPU", "devices": "0", "gpu_ram_part": .35,
            "gpu_cat_features_storage": "CpuPinnedMemory", "pinned_memory_size": "2gb",
            "max_ctr_complexity": 1, "border_count": 128, "boosting_type": "Plain",
            "l2_leaf_reg": 30 if "smooth" in name else 15,
            "allow_writing_files": False}


def gpu_status() -> dict[str, Any]:
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    parts = raw.splitlines()[0].split(",")
    return {"name": parts[0].strip(), "total_mib": int(parts[1]), "used_mib": int(parts[2]),
            "free_mib": int(parts[3]), "utilization_percent": int(parts[4])}


def fit_gpu(x: pd.DataFrame, y: pd.Series[Any], spec: ModelSpec, extra: dict[str, Any]) -> NovelGuardedForecast:
    """Reuse unchanged fit-only vocabulary/baseline/native artifact and guard policy."""
    model = NovelGuardedForecast(spec.model_copy(update={"kind": "mean"})).fit(x, y)
    model.spec = spec
    model.estimator = CatBoostRegressor(iterations=spec.iterations, depth=spec.depth,
        learning_rate=spec.learning_rate, loss_function="RMSE", random_seed=spec.seed,
        thread_count=spec.threads, verbose=500, **extra)
    model.estimator.fit(model.encode(x), y, cat_features=CATEGORIES)
    return model


def validate_inputs(data: pd.DataFrame, x: pd.DataFrame) -> None:
    if len(data) != len(x) or not data.index.equals(x.index) or data[ID].duplicated().any():
        raise ValueError("Prepared data/features have different identities or order")
    clock = pd.to_datetime(data[TIME], utc=True)
    if not clock.lt(pd.Timestamp("2025-12-01T00:00:00Z")).all():
        raise ValueError("GPU search input must exclude December and all 2026 rows")
    if not np.isfinite(data[TARGET]).all():
        raise ValueError("Nonfinite labels")
    allowed = set(CATEGORIES) | {"hour_utc", "weekday_utc", "month_utc", "weekend", "hour_sin", "hour_cos", "year_sin", "year_cos"}
    allowed |= {f"scheduled_{phase}_{window}" for phase in ["dep", "arr"] for window in ["pm5m", "pm10m", "pm15m", "pm30m", "pm60m", "previous30m"]}
    if set(x.columns) - allowed:
        raise ValueError("Features outside the frozen forecast allowlist")


def search(root: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Use a new GPU output directory")
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    for name, digest in plan["source_hashes"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError("Frozen GPU source changed")
    for name, digest in plan["input_hashes"].items():
        if sha256(root / name) != digest:
            raise ValueError("Prepared GPU inputs changed")
    expected = {n: s.model_dump() for n, s in candidates().items()}
    if plan["models"] != expected or plan["training_overrides"] != {n: overrides(n) for n in expected}:
        raise ValueError("GPU candidates changed after registration")
    data, x = pd.read_parquet(root / "departures.parquet"), pd.read_parquet(root / "features.parquet")
    validate_inputs(data, x)
    output.mkdir()
    write_json(output / "plan.json", plan)
    write_json(output / "leaderboard-provenance.json", plan["git_provenance"])
    write_json(output / "environment.json", {"source_hashes": plan["source_hashes"], "gpu_initial": gpu_status(),
        "python": subprocess.check_output(["python3", "--version"], text=True).strip(),
        "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        "gpu_training_determinism": "GPU floating-point accumulation is nondeterministic; exact saved-model inference is verified"})
    ledger: list[dict[str, Any]] = []
    all_folds = []
    for month in [9, 10, 11]:
        fit, valid, purged = purged_split(data, month)
        result = data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
        for name, spec in candidates().items():
            hardware = gpu_status()
            if hardware["free_mib"] < 2600:
                raise RuntimeError("Insufficient idle GPU memory; preserve existing jobs")
            stamp = datetime.now(timezone.utc).isoformat()
            print(stamp, "FIT", month, name, hardware, flush=True)
            before = time.monotonic()
            model = fit_gpu(x.loc[fit], data.loc[fit, TARGET], spec, overrides(name))
            fit_seconds = time.monotonic() - before
            before_predict = time.monotonic()
            prediction = model.predict(x.loc[valid])
            inference_seconds = time.monotonic() - before_predict
            model_root = output / "models" / f"{month:02}" / name
            model_root.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_root)
            reloaded = load_model(model_root)
            if not np.array_equal(prediction, reloaded.predict(x.loc[valid])):
                raise ValueError("Saved GPU model does not reload to exact CPU predictions")
            result[name] = prediction
            report = metrics(result[TARGET], prediction, result["ADEP_mvt"])
            ledger.append({"experiment": f"forward-{month:02}-{name}", "model": name, "device": "GPU RTX4090 / CPU inference",
                "parameters": spec.model_dump(), "gpu_parameters": overrides(name), "features": model.columns,
                "training_rows": int(fit.sum()), "purged_flight_rows": purged, "validation_month": month,
                "training_seconds": fit_seconds + inference_seconds, "training_time_seconds": fit_seconds,
                "inference_time_seconds": inference_seconds, "native_reload_exact_rows": len(result),
                "started_at_utc": stamp, "completed_at_utc": datetime.now(timezone.utc).isoformat(), **report})
            temporary = output / f"selection-{month:02}.parquet.tmp"
            result.to_parquet(temporary, index=False)
            os.replace(temporary, output / f"selection-{month:02}.parquet")
            write_json(output / "experiment-ledger.json", ledger)
            print("PRC_EVENT " + json.dumps({"month": month, "model": name, "rmse": report["rmse"], "mae": report["mae"],
                "prediction_sha256": sha256(output / f"selection-{month:02}.parquet"),
                "ledger_sha256": sha256(output / "experiment-ledger.json")}), flush=True)
            del model, reloaded
            gc.collect()
        all_folds.append(result)
    pooled = pd.concat(all_folds, ignore_index=True)
    pooled.to_parquet(output / "selection-oof.parquet", index=False)
    write_json(output / "report.json", {"status": "COMPLETED_FORWARD_GPU_SEARCH_NO_DECEMBER_OR_OFFICIAL_FEEDBACK",
        "scores": {n: metrics(pooled[TARGET], pooled[n], pooled["ADEP_mvt"]) for n in candidates()}, "gpu_final": gpu_status()})
    print("PRC_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    search(args.inputs, args.output)
