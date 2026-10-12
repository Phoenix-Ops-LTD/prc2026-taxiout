# SPDX-License-Identifier: GPL-3.0-only
"""Frozen airport CatBoost specialists with native artifacts and forward-only CV."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, ConfigDict, Field

from forecast_features import CATEGORIES, metrics, purged_split
from forecast_gpu_compare import evaluate
from forecast_gpu_search import gpu_status, overrides, validate_inputs
from forecast_model import FittedForecast, ModelSpec
from forecast_novel_guard import NovelGuardedForecast, load_model
from pipeline import ID, TARGET, TIME, sha256, write_json

POLICY = "airport-catboost-novel-aircraft/1.0"


class SpecialistSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["catboost"] = "catboost"
    global_model: str
    iterations: int = Field(ge=1, le=5000)
    depth: int = Field(ge=2, le=10)
    learning_rate: float = Field(default=.04, gt=0, le=1)
    demand: bool
    threads: int = 2
    seed: int = 20261011
    minimum_airport_rows: int = 10000
    local_weight: float = Field(default=.5, ge=.5, le=.5)


def candidates() -> dict[str, SpecialistSpec]:
    return {
        "gpu_airport_no_demand_novel_guarded": SpecialistSpec(global_model="gpu_deep_no_demand_novel_guarded", iterations=1200, depth=6, demand=False),
        "gpu_airport_demand_novel_guarded": SpecialistSpec(global_model="gpu_deep_demand_novel_guarded", iterations=1200, depth=6, demand=True),
        "gpu_airport_deep_demand_novel_guarded": SpecialistSpec(global_model="gpu_deep_demand_novel_guarded", iterations=800, depth=8, demand=True),
    }


class AirportCatBoostForecast(NovelGuardedForecast):
    def save(self, root: Path) -> dict[str, Any]:
        metadata = super().save(root)
        files = {}
        for i, (airport, model) in enumerate(sorted(self.specialists.items())):
            name = f"airport-{i}.cbm"
            model.save_model(str(root / name))
            files[airport] = name
        metadata.update(guard_policy=POLICY, specialist_files=files, local_weight=.5,
            artifacts={p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != "metadata.json"})
        write_json(root / "metadata.json", metadata)
        return metadata

    @classmethod
    def load(cls, root: Path) -> AirportCatBoostForecast:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("guard_policy") != POLICY or metadata.get("local_weight") != .5 or not isinstance(metadata.get("missing_aircraft_seen_in_fit"), bool):
            raise ValueError("Airport specialist policy mismatch")
        base = FittedForecast.load(root)
        obj = cls(base.spec)
        obj.__dict__.update(base.__dict__)
        obj.missing_aircraft_seen_in_fit = metadata["missing_aircraft_seen_in_fit"]
        for airport, name in metadata["specialist_files"].items():
            if name not in metadata["artifacts"] or Path(name).name != name:
                raise ValueError("Unbound airport artifact")
            local = CatBoostRegressor()
            local.load_model(str(root / name))
            obj.specialists[airport] = local
        return obj


def sources() -> list[str]:
    return ["forecast_gpu_airport.py", "forecast_gpu_search.py", "forecast_gpu_compare.py", "forecast_features.py", "forecast_model.py", "forecast_guard.py", "forecast_novel_guard.py", "pipeline.py"]


def freeze(inputs: Path, checkpoints: Path, output: Path, commit: str) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Frozen plan must be new")
    original = json.loads((inputs / "plan.json").read_text(encoding="utf-8"))
    for name, digest in original["source_hashes"].items():
        if sha256(inputs / name) != digest:
            raise ValueError("Original frozen source changed")
    plan = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection_months": [9, 10, 11], "execution_unit": "GPU",
        "models": {n: s.model_dump() for n, s in candidates().items()},
        "training_overrides": {n: overrides("airport") for n in candidates()},
        "source_hashes": {n: sha256(inputs / n) for n in sources()}, "input_hashes": original["input_hashes"],
        "training_input_hashes": original["training_input_hashes"], "base_feature_sha256": original["base_feature_sha256"],
        "checkpoint_hashes": {str(p.relative_to(checkpoints)): sha256(p) for m in [9, 10, 11] for n in sorted({s.global_model for s in candidates().values()}) for p in sorted((checkpoints / "models" / f"{m:02}" / n).iterdir()) if p.is_file()},
        "checkpoint_ledger_sha256": sha256(checkpoints / "experiment-ledger.json"),
        "git_provenance": {"git_commit": commit, "git_dirty_at_training": True, "commit_provenance": "EXACT_HEAD_AT_AIRPORT_PLAN_FREEZE; source hashes bind the new specialist implementation"},
        "official_feedback_used": False, "december_inputs_included": False, "previous_diagnostics_observed": True,
        "selection_rule": "Same complete purged forward folds; fixed equal pairs with incumbent components; >=1s pooled RMSE gain and every-fold improvement required",
        "memory_policy": "35% of free memory; require >=2600 MiB before each airport fit; existing jobs preserved",
        "gpu_training_determinism": "Nondeterministic training; exact saved native CPU inference verified"}
    write_json(output, plan)
    return plan


def train(inputs: Path, checkpoints: Path, plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Use a new specialist output directory")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["models"] != {n: s.model_dump() for n, s in candidates().items()} or plan["training_overrides"] != {n: overrides("airport") for n in candidates()}:
        raise ValueError("Frozen airport parameters changed")
    for binding_root, key in [(inputs, "source_hashes"), (inputs, "input_hashes"), (checkpoints, "checkpoint_hashes")]:
        for name, digest in plan[key].items():
            if sha256(binding_root / name) != digest:
                raise ValueError("Frozen input, source or checkpoint changed")
    if sha256(checkpoints / "experiment-ledger.json") != plan["checkpoint_ledger_sha256"]:
        raise ValueError("Global checkpoint ledger changed")
    original_ledger = json.loads((checkpoints / "experiment-ledger.json").read_text(encoding="utf-8"))
    data, x = pd.read_parquet(inputs / "departures.parquet"), pd.read_parquet(inputs / "features.parquet")
    validate_inputs(data, x)
    output.mkdir()
    write_json(output / "plan.json", plan)
    write_json(output / "leaderboard-provenance.json", plan["git_provenance"])
    write_json(output / "environment.json", {"source_hashes": plan["source_hashes"], "gpu_initial": gpu_status(),
        "python": sys.version, "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)})
    ledger: list[dict[str, Any]] = []
    folds = []
    for month in [9, 10, 11]:
        fit, valid, purged = purged_split(data, month)
        result = data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
        for name, spec in candidates().items():
            base_records = [r for r in original_ledger if r["model"] == spec.global_model and r["validation_month"] == month]
            if len(base_records) != 1 or base_records[0]["training_rows"] != int(fit.sum()):
                raise ValueError("Global checkpoint does not match the fitting fold")
            base = load_model(checkpoints / "models" / f"{month:02}" / spec.global_model)
            if not isinstance(base, NovelGuardedForecast) or base.spec.demand != spec.demand:
                raise ValueError("Global checkpoint policy mismatch")
            model = AirportCatBoostForecast(base.spec)
            model.__dict__.update(base.__dict__)
            encoded = model.encode(x.loc[fit])
            stamp = datetime.now(timezone.utc).isoformat()
            print(stamp, "FIT", month, name, flush=True)
            start = time.monotonic()
            local = None
            for airport in sorted(x.loc[fit, "ADEP_mvt"].unique()):
                mask = x.loc[fit, "ADEP_mvt"].eq(airport)
                if int(mask.sum()) < spec.minimum_airport_rows:
                    continue
                if gpu_status()["free_mib"] < 2600:
                    raise RuntimeError("Insufficient idle memory; preserve existing jobs")
                local = CatBoostRegressor(iterations=spec.iterations, depth=spec.depth, learning_rate=spec.learning_rate,
                    loss_function="RMSE", random_seed=spec.seed, thread_count=spec.threads, verbose=False, **overrides("airport"))
                local.fit(encoded.loc[mask], data.loc[fit, TARGET].loc[mask], cat_features=CATEGORIES)
                model.specialists[str(airport)] = local
                print("AIRPORT_READY", month, name, airport, int(mask.sum()), flush=True)
            fit_seconds = time.monotonic() - start
            start = time.monotonic()
            prediction = model.predict(x.loc[valid])
            inference_seconds = time.monotonic() - start
            root = output / "models" / f"{month:02}" / name
            root.parent.mkdir(parents=True, exist_ok=True)
            metadata = model.save(root)
            if not np.array_equal(prediction, AirportCatBoostForecast.load(root).predict(x.loc[valid])):
                raise ValueError("Saved airport native model parity failed")
            result[name] = prediction
            report = metrics(result[TARGET], prediction, result["ADEP_mvt"])
            ledger.append({"experiment": f"forward-{month:02}-{name}", "model": name, "parameters": spec.model_dump(),
                "features": model.columns, "training_rows": int(fit.sum()), "purged_flight_rows": purged, "validation_month": month,
                "training_seconds": fit_seconds + base_records[0]["training_time_seconds"] + inference_seconds,
                "training_time_seconds": fit_seconds + base_records[0]["training_time_seconds"],
                "incremental_training_time_seconds": fit_seconds,
                "inference_time_seconds": inference_seconds, "training_time_scope": "TOTAL_MEASURED_GLOBAL_PLUS_SPECIALIST_FITS; global checkpoint reused",
                "reused_global_training_seconds": base_records[0]["training_time_seconds"], "device": "GPU RTX4090 / CPU inference",
                "native_reload_exact_rows": len(result), "native_artifact_hashes": metadata["artifacts"],
                "started_at_utc": stamp, "completed_at_utc": datetime.now(timezone.utc).isoformat(), **report})
            path = output / f"selection-{month:02}.parquet"
            temporary = path.with_suffix(".parquet.tmp")
            result.to_parquet(temporary, index=False)
            os.replace(temporary, path)
            write_json(output / "experiment-ledger.json", ledger)
            snapshot = output / "snapshots" / f"{month:02}-{name}"
            snapshot.mkdir(parents=True)
            shutil.copyfile(path, snapshot / path.name)
            shutil.copyfile(output / "experiment-ledger.json", snapshot / "experiment-ledger.json")
            print("PRC_EVENT " + json.dumps({"month": month, "model": name, "snapshot": str(snapshot.relative_to(output)),
                "prediction_sha256": sha256(snapshot / path.name), "ledger_sha256": sha256(snapshot / "experiment-ledger.json")}), flush=True)
            del model, base, local, encoded
            gc.collect()
        folds.append(result)
    pooled = pd.concat(folds, ignore_index=True)
    pooled.to_parquet(output / "selection-oof.parquet", index=False)
    write_json(output / "report.json", {"status": "COMPLETED_FORWARD_GPU_SEARCH_AIRPORT_SPECIALISTS_NO_LATER_FEEDBACK",
        "scores": {n: metrics(pooled[TARGET], pooled[n], pooled["ADEP_mvt"]) for n in candidates()}, "gpu_final": gpu_status()})
    print("PRC_COMPLETE", flush=True)


def compare(gpu: Path, incumbent: Path, output: Path) -> dict[str, Any]:
    if output.exists() or not (gpu / "report.json").exists():
        raise ValueError("Complete batch and new comparison directory required")
    a, b = pd.read_parquet(incumbent / "selection-oof.parquet"), pd.read_parquet(gpu / "selection-oof.parquet")
    keys = [ID, TARGET, TIME, "ADEP_mvt"]
    if not a[keys].equals(b[keys]):
        raise ValueError("Forward identities, labels or order differ")
    old = json.loads((incumbent / "selection.json").read_text(encoding="utf-8"))
    weights = old["scores"][old["selected"]]["weights"]
    frame = a[keys + list(weights)].copy()
    for name in candidates():
        frame[name] = b[name]
    result = evaluate(frame, weights, list(candidates()))
    if not np.isclose(result["incumbent"]["rmse"], old["scores"][old["selected"]]["rmse"], rtol=0, atol=1e-8):
        raise ValueError("Incumbent score changed")
    output.mkdir()
    write_json(output / "plan.json", {"selection_months": [9, 10, 11], "source_sha256": sha256(Path(__file__)),
        "rule": result["rule"], "input_hashes": {"gpu_oof": sha256(gpu / "selection-oof.parquet"), "incumbent_oof": sha256(incumbent / "selection-oof.parquet")}})
    frame.to_parquet(output / "selection-oof.parquet", index=False)
    result["selected_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "selection.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    for name in ["inputs", "checkpoints", "output"]:
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--git-commit", required=True)
    runner = commands.add_parser("train")
    for name in ["inputs", "checkpoints", "plan", "output"]:
        runner.add_argument("--" + name, type=Path, required=True)
    comparison = commands.add_parser("compare")
    for name in ["gpu", "incumbent", "output"]:
        comparison.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.inputs, args.checkpoints, args.output, args.git_commit), indent=2))
    elif args.command == "train":
        train(args.inputs, args.checkpoints, args.plan, args.output)
    else:
        result = compare(args.gpu, args.incumbent, args.output)
        print(json.dumps({"selected": result["selected"], "promoted": result["promoted"], "best_new_rmse": min(s["rmse"] for s in result["scores"].values())}))
