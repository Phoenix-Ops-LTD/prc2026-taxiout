# SPDX-License-Identifier: GPL-3.0-only
"""Fixed ordinary-scope airport and NM-neighbor ensemble for independent research."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from arrival_contest import matrix as arrival_matrix
from carrier_contest import specialist_scope
from ensemble_contest import categorical
from nm_neighbor_features import completed_nm_neighbors
from pipeline import ID, TARGET, sha256, validate_submission, write_json
from traffic_features import nm_schedule_fallback_baseline


class OrdinaryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: Literal["prc2026-ordinary-ensemble/9.0.0"] = "prc2026-ordinary-ensemble/9.0.0"
    trees: int = Field(default=679, ge=1, le=679)
    data_permission_ref: str = Field(min_length=1, pattern=r"\S")


def parameters(config: OrdinaryConfig) -> dict[str, Any]:
    return dict(n_estimators=config.trees, objective="regression", num_leaves=63, learning_rate=.035,
        min_child_samples=80, reg_lambda=10., colsample_bytree=.9, n_jobs=2, verbosity=-1,
        random_state=20260907, deterministic=True, force_col_wise=True)


def matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dep, x = arrival_matrix(raw)
    columns = [str(c) for c in x.columns]
    extra = completed_nm_neighbors(dep, raw)
    if not extra.index.equals(x.index):
        raise ValueError("Neighbor feature alignment differs")
    return dep, pd.concat([x, extra], axis=1), columns


def blend(x: pd.DataFrame, baseline: pd.Series, neighbor: pd.Series, airport: pd.Series) -> pd.Series:
    for values in [baseline, neighbor, airport]:
        if not values.index.equals(x.index):
            raise ValueError("Prediction row alignment differs")
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError("Predictions must be finite and nonnegative")
    candidate = .5 * airport + .5 * neighbor
    result = .75 * baseline + .25 * candidate
    scope = specialist_scope(x)
    result.loc[scope] = baseline.loc[scope]
    return result


def train(data: Path, output: Path, config: OrdinaryConfig) -> None:
    if output.exists():
        raise ValueError("Use a fresh immutable run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    hashes = {p.name: sha256(p) for p in paths}
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    clock = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True, errors="raise")
    if clock.isna().any() or not clock.dt.year.eq(2025).all():
        raise ValueError("Complete 2025 movement timestamps are required")
    print("BUILDING_ORDINARY_FEATURES", len(raw), flush=True)
    dep, x, columns = matrix(raw)
    del raw
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Labels must be finite")
    fit = y.ge(0) & ~specialist_scope(x)
    if not fit.any():
        raise ValueError("Ordinary-scope fitting rows are required")
    offset = nm_schedule_fallback_baseline(x)
    cats = [str(c) for c in x.select_dtypes(include=["object", "str"]).columns]
    x = categorical(x)
    params = parameters(config)
    output.mkdir(parents=True)
    write_json(output / "inputs.json", {"status": "FULL_FIT_RUNNING_NOT_OFFICIAL", "config": config.model_dump(),
        "parameters": params, "airport_features": columns, "neighbor_features": list(x),
        "categories": cats, "input_hashes": hashes, "fit_rows": int(fit.sum()), "residual_fit_cap": 7200})
    print("FITTING_FULL_NEIGHBOR", int(fit.sum()), flush=True)
    model = lgb.LGBMRegressor(**params)
    model.fit(x.loc[fit], (y-offset).clip(-7200, 7200).loc[fit], categorical_feature=cats)
    model.booster_.save_model(str(output / "neighbor-model.txt"))
    digests = {"neighbor": sha256(output / "neighbor-model.txt")}
    counts: dict[str, int] = {}
    for airport in sorted(dep.loc[fit, "ADEP_mvt"].unique()):
        if not isinstance(airport, str) or len(airport) != 4 or not airport.isalpha():
            raise ValueError("Airport names must be four-letter ICAO codes")
        scope = fit & dep["ADEP_mvt"].eq(airport)
        counts[airport] = int(scope.sum())
        print("FITTING_FULL_AIRPORT", airport, counts[airport], flush=True)
        expert = lgb.LGBMRegressor(**params)
        expert.fit(x.loc[scope, columns], (y-offset).clip(-7200, 7200).loc[scope], categorical_feature=cats)
        path = output / f"airport-{airport}.txt"
        expert.booster_.save_model(str(path))
        digests[airport] = sha256(path)
        write_json(output / "completed-models.json", digests)
    write_json(output / "report.json", {"status": "FULL_FIT_NOT_OFFICIAL_SCORE", "config": config.model_dump(),
        "parameters": params, "airport_features": columns, "neighbor_features": list(x), "categories": cats,
        "fit_rows": int(fit.sum()), "airport_fit_rows": counts, "input_hashes": hashes,
        "residual_fit_cap": 7200, "model_sha256": digests, "candidate_weight": .25,
        "neighbor_weight": .5, "airport_weight": .5})
    print("ORDINARY_ENSEMBLE_READY", flush=True)


def checked_model(path: Path, digest: str, columns: list[str], trees: int) -> Any:
    if sha256(path) != digest:
        raise ValueError("Model digest mismatch")
    model = lgb.Booster(model_file=str(path))
    if model.feature_name() != columns or model.current_iteration() != trees:
        raise ValueError("Stored model feature order or tree count differs")
    return model


def predict(run: Path, baseline_path: Path, ranking: Path, template_path: Path,
            output: Path, permission: str) -> None:
    if output.exists() or output.with_suffix(".manifest.json").exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    config = OrdinaryConfig(trees=report["parameters"]["n_estimators"], data_permission_ref=permission)
    if report.get("status") != "FULL_FIT_NOT_OFFICIAL_SCORE" or report["parameters"] != parameters(config):
        raise ValueError("A completed full fit with the documented parameters is required")
    for key, value in {"residual_fit_cap": 7200, "candidate_weight": .25, "neighbor_weight": .5, "airport_weight": .5}.items():
        if report.get(key) != value:
            raise ValueError("Stored configuration differs from the fixed ensemble")
    manifest = json.loads(baseline_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_version") != "prc2026-arrival-boost-blend/8.0.0":
        raise ValueError("The documented v8 baseline is required")
    if manifest["submission_sha256"] != sha256(baseline_path):
        raise ValueError("Baseline digest mismatch")
    if manifest["ranking_sha256"] != sha256(ranking) or manifest["template_sha256"] != sha256(template_path):
        raise ValueError("Baseline data or template digest mismatch")
    template = pd.read_parquet(template_path)
    baseline_frame = pd.read_parquet(baseline_path)
    validate_submission(template, baseline_frame)
    dep, x, columns = matrix(pd.read_parquet(ranking))
    if report["airport_features"] != columns or report["neighbor_features"] != list(x):
        raise ValueError("Feature schema differs from the saved report")
    offset = nm_schedule_fallback_baseline(x)
    cats = categorical(x)
    neighbor_model = checked_model(run / "neighbor-model.txt", report["model_sha256"]["neighbor"],
        [str(c) for c in x.columns], config.trees)
    neighbor = pd.Series(np.maximum(0, neighbor_model.predict(cats, num_threads=2) + offset), index=x.index)
    baseline = pd.Series(dep[ID].map(baseline_frame.set_index(ID)[TARGET]).to_numpy(), index=x.index)
    airport_values = baseline.copy()
    ordinary = ~specialist_scope(x)
    for airport in sorted(dep.loc[ordinary, "ADEP_mvt"].unique()):
        if airport not in report["airport_fit_rows"] or not isinstance(airport, str) or len(airport) != 4 or not airport.isalpha():
            raise ValueError("No verified expert exists for a queried airport")
        scope = ordinary & dep["ADEP_mvt"].eq(airport)
        expert = checked_model(run / f"airport-{airport}.txt", report["model_sha256"][airport], columns, config.trees)
        airport_values.loc[scope] = np.maximum(0, expert.predict(cats.loc[scope, columns], num_threads=2) + offset.loc[scope])
    values = blend(x, baseline, neighbor, airport_values)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    write_json(output.with_suffix(".manifest.json"), {"data_class": "competition", "data_permission_ref": permission,
        "model_version": config.model_version, "submission_sha256": sha256(output), "template_sha256": sha256(template_path),
        "ranking_sha256": sha256(ranking), "baseline_sha256": sha256(baseline_path), "baseline_model_version": manifest["model_version"],
        "model_sha256": report["model_sha256"], "candidate_weight": .25, "neighbor_weight": .5, "airport_weight": .5,
        "residual_fit_cap": 7200, "rows": len(result), "specialist_rows": int((~ordinary).sum())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for name in ["data", "output"]:
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--permission-ref", required=True)
    fit.add_argument("--trees", type=int, default=679)
    pred = commands.add_parser("predict")
    for name in ["run", "baseline", "ranking", "template", "output"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, OrdinaryConfig(trees=args.trees, data_permission_ref=args.permission_ref))
    else:
        predict(args.run, args.baseline, args.ranking, args.template, args.output, args.permission_ref)


if __name__ == "__main__":
    main()
