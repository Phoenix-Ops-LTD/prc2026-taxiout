# SPDX-License-Identifier: GPL-3.0-only
"""Resume a frozen GPU batch and publish immutable per-experiment transfer snapshots."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_features import metrics, purged_split
from forecast_gpu_search import candidates, fit_gpu, gpu_status, overrides, validate_inputs
from forecast_novel_guard import load_model
from pipeline import ID, TARGET, TIME, sha256, write_json


def resume(inputs: Path, output: Path) -> None:
    plan = json.loads((inputs / "plan.json").read_text(encoding="utf-8"))
    if plan != json.loads((output / "plan.json").read_text(encoding="utf-8")):
        raise ValueError("Resume plan differs from the frozen experiment")
    for name, digest in plan["source_hashes"].items():
        if sha256(inputs / name) != digest:
            raise ValueError("Frozen training source changed")
    for name, digest in plan["input_hashes"].items():
        if sha256(inputs / name) != digest:
            raise ValueError("Frozen input changed")
    if plan["models"] != {n: s.model_dump() for n, s in candidates().items()} or plan["training_overrides"] != {n: overrides(n) for n in candidates()}:
        raise ValueError("Resume parameters changed")
    if (output / "report.json").exists():
        raise ValueError("Completed batches do not need resuming")
    data = pd.read_parquet(inputs / "departures.parquet")
    x = pd.read_parquet(inputs / "features.parquet")
    validate_inputs(data, x)
    ledger = json.loads((output / "experiment-ledger.json").read_text(encoding="utf-8"))
    completed = {(r["validation_month"], r["model"]) for r in ledger}
    if len(completed) != len(ledger) or not completed <= {(m, n) for m in [9, 10, 11] for n in candidates()}:
        raise ValueError("Unexpected or duplicate completed experiment")
    write_json(output / "resume-receipt.json", {"resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_source_sha256": sha256(Path(__file__)), "completed_experiments_preserved": len(ledger),
        "frozen_fit_source_unchanged": True, "parameters_unchanged": True})
    folds = []
    for month in [9, 10, 11]:
        fit, valid, purged = purged_split(data, month)
        path = output / f"selection-{month:02}.parquet"
        result = pd.read_parquet(path) if path.exists() else data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
        expected = data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].reset_index(drop=True)
        if not result[expected.columns].reset_index(drop=True).equals(expected):
            raise ValueError("Existing fold identities, labels or order changed")
        for name, spec in candidates().items():
            model_root = output / "models" / f"{month:02}" / name
            if (month, name) in completed:
                if name not in result or not np.array_equal(result[name].to_numpy(), load_model(model_root).predict(x.loc[valid])):
                    raise ValueError("Completed native model no longer matches its prediction receipt")
                continue
            hardware = gpu_status()
            if hardware["free_mib"] < 2600:
                raise RuntimeError("Insufficient idle GPU memory; preserve existing jobs")
            if model_root.exists():
                raise ValueError("Unrecorded native model requires explicit audit")
            stamp = datetime.now(timezone.utc).isoformat()
            print(stamp, "FIT", month, name, hardware, flush=True)
            start = time.monotonic()
            model = fit_gpu(x.loc[fit], data.loc[fit, TARGET], spec, overrides(name))
            fit_seconds = time.monotonic() - start
            start = time.monotonic()
            prediction = model.predict(x.loc[valid])
            inference_seconds = time.monotonic() - start
            model_root.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_root)
            if not np.array_equal(prediction, load_model(model_root).predict(x.loc[valid])):
                raise ValueError("Native reload differs from GPU-trained model inference")
            result[name] = prediction
            report = metrics(result[TARGET], prediction, result["ADEP_mvt"])
            ledger.append({"experiment": f"forward-{month:02}-{name}", "model": name,
                "device": "GPU RTX4090 / CPU inference", "parameters": spec.model_dump(), "gpu_parameters": overrides(name),
                "features": model.columns, "training_rows": int(fit.sum()), "purged_flight_rows": purged,
                "validation_month": month, "training_seconds": fit_seconds + inference_seconds,
                "training_time_seconds": fit_seconds, "inference_time_seconds": inference_seconds,
                "native_reload_exact_rows": len(result), "started_at_utc": stamp,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(), **report})
            result.to_parquet(path, index=False)
            write_json(output / "experiment-ledger.json", ledger)
            snapshot = output / "snapshots" / f"{month:02}-{name}"
            snapshot.mkdir(parents=True)
            shutil.copyfile(path, snapshot / path.name)
            shutil.copyfile(output / "experiment-ledger.json", snapshot / "experiment-ledger.json")
            print("PRC_EVENT " + json.dumps({"month": month, "model": name, "snapshot": str(snapshot.relative_to(output)),
                "prediction_sha256": sha256(snapshot / path.name), "ledger_sha256": sha256(snapshot / "experiment-ledger.json")}), flush=True)
            del model
            gc.collect()
        folds.append(result)
    pooled = pd.concat(folds, ignore_index=True)
    pooled.to_parquet(output / "selection-oof.parquet", index=False)
    write_json(output / "report.json", {"status": "COMPLETED_FORWARD_GPU_SEARCH_NO_DECEMBER_OR_OFFICIAL_FEEDBACK",
        "scores": {n: metrics(pooled[TARGET], pooled[n], pooled["ADEP_mvt"]) for n in candidates()}, "gpu_final": gpu_status()})
    print("PRC_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resume(args.inputs, args.output)
