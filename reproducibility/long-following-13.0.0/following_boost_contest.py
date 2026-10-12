# SPDX-License-Identifier: GPL-3.0-only
"""Fixed Cat183 replacement of v11, preserving its specialist and clock values.

Retrospective PRC research. Fitting and prediction do not select or upload a
submission, and do not establish an official improvement.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, ConfigDict, Field, model_validator

import nm_clock_overlay
from carrier_contest import specialist_scope
from neighbor_boost_contest import (
    NeighborBoostConfig, checked_baseline, checked_boost, parameters as neighbor_parameters, read_object,
)
from nm_clock_overlay import apply_clock_overlay
from nm_following_features import following_nm_neighbors
from ordinary_ensemble import matrix as neighbor_matrix
from pipeline import ID, TARGET, sha256, validate_submission, write_json
from traffic_features import nm_schedule_fallback_baseline

VERSION = "prc2026-following-boost-replacement/12.0.0"
V11_VERSION = "prc2026-nm-clock-overlay/11.0.0"
V10_VERSION = "prc2026-neighbor-boost-replacement/10.0.0"
REPLACEMENT_WEIGHT = .375
RESIDUAL_CAP = 7200


class FollowingBoostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: Literal["prc2026-following-boost-replacement/12.0.0"] = "prc2026-following-boost-replacement/12.0.0"
    trees: int = Field(default=4999, ge=1, le=4999)
    device: Literal["CPU", "GPU"] = "GPU"
    data_class: Literal["competition", "synthetic"] = "competition"
    data_permission_ref: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def frozen_competition_fit(self) -> Self:
        if self.data_class == "competition" and (self.trees != 4999 or self.device != "GPU"):
            raise ValueError("Competition fitting requires the frozen 4999 GPU trees")
        return self


def shared_configuration(config: FollowingBoostConfig) -> NeighborBoostConfig:
    """Reuse the incumbent component's unchanged numerical-parameter checks."""
    return NeighborBoostConfig(trees=config.trees, device=config.device, data_class=config.data_class,
        data_permission_ref=config.data_permission_ref)


def parameters(config: FollowingBoostConfig, output: Path) -> dict[str, Any]:
    return neighbor_parameters(shared_configuration(config), output)


def matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dep, x, _ = neighbor_matrix(raw)
    old_columns = [str(column) for column in x.columns]
    extra = following_nm_neighbors(dep, raw)
    if not extra.index.equals(x.index) or len(old_columns) != 153 or extra.shape[1] != 30:
        raise ValueError("Expected aligned 153 incumbent and 30 following-context features")
    return dep, pd.concat([x, extra], axis=1), old_columns


def clock_scope(dep: pd.DataFrame) -> pd.Series:
    """The frozen rule always replaces eligible rows with a strictly positive value.

    Its actual proxy is at least -7200, while AOBT minus LOBT is strictly
    greater than 7200. Thus applying it to zeros identifies its exact scope.
    """
    rows = dep.set_index(ID)
    probe = apply_clock_overlay(rows, pd.Series(0., index=rows.index))
    return pd.Series(probe.gt(0).to_numpy(), index=dep.index)


def replace_component(dep: pd.DataFrame, x: pd.DataFrame, baseline: pd.Series,
        new: pd.Series, old: pd.Series) -> pd.Series:
    if not dep.index.equals(x.index):
        raise ValueError("Movement and feature row alignment differs")
    for values in [baseline, new, old]:
        if not values.index.equals(x.index):
            raise ValueError("Prediction row alignment differs")
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError("Component predictions must be finite and nonnegative")
    protected = specialist_scope(x) | clock_scope(dep)
    result = baseline.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        result.loc[~protected] = baseline.loc[~protected] + REPLACEMENT_WEIGHT * (new.loc[~protected] - old.loc[~protected])
    if not np.isfinite(result).all() or result.lt(0).any():
        raise ValueError("Replacement must be finite and nonnegative; do not clip")
    if not np.array_equal(result.loc[protected], baseline.loc[protected]):
        raise ValueError("Specialist and clock predictions must remain exact")
    return result


def train(data: Path, output: Path, config: FollowingBoostConfig) -> None:
    if output.exists():
        raise ValueError("Use a fresh immutable run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    hashes = {path.name: sha256(path) for path in paths}
    source = Path(__file__).resolve().parent
    sources = {path.name: sha256(path) for path in source.glob("*.py")}
    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    clock = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True, errors="raise")
    if clock.isna().any() or not clock.dt.year.eq(2025).all() or set(clock.dt.month) != set(range(1, 13)):
        raise ValueError("Complete movement clocks covering all twelve 2025 months are required")
    print("BUILDING_FOLLOWING_BOOST_FEATURES", len(raw), flush=True)
    dep, x, old_columns = matrix(raw)
    del raw
    if len(x.columns) != 183 or len(old_columns) != 153:
        raise ValueError("Expected the documented 183 and 153 feature schemas")
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Labels must be finite")
    fit = y.ge(0) & ~specialist_scope(x)
    if not fit.any():
        raise ValueError("Ordinary-scope fitting rows are required")
    offset = nm_schedule_fallback_baseline(x)
    cats = [str(column) for column in x.select_dtypes(include=["str", "object"]).columns]
    params = parameters(config, output)
    if {path.name: sha256(path) for path in paths} != hashes or any(sha256(source / name) != digest for name, digest in sources.items()):
        raise ValueError("Training input or source changed while building features")
    output.mkdir(parents=True)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name in sources:
        shutil.copyfile(source / name, snapshot / name)
    metadata = {"config": config.model_dump(), "parameters": params, "features": list(x),
        "categories": cats, "old_features": old_columns, "input_hashes": hashes, "source_sha256": sources,
        "departure_rows": len(dep), "fit_rows": int(fit.sum()), "negative_label_rows": int(y.lt(0).sum()),
        "capped_fit_rows": int((y-offset).loc[fit].abs().gt(RESIDUAL_CAP).sum()),
        "residual_cap": RESIDUAL_CAP, "replacement_weight": REPLACEMENT_WEIGHT,
        "clock_rule_version": nm_clock_overlay.VERSION}
    write_json(output / "inputs.json", metadata)
    model = CatBoostRegressor(**params)
    print("FITTING_FOLLOWING_BOOST", int(fit.sum()), config.trees, flush=True)
    model.fit(x.loc[fit], (y-offset).clip(-RESIDUAL_CAP, RESIDUAL_CAP).loc[fit], cat_features=cats,
        save_snapshot=True, snapshot_file=str((output / "training.snapshot").resolve()), snapshot_interval=60)
    if model.tree_count_ != config.trees:
        raise ValueError("Fitted tree count differs from the frozen configuration")
    if {path.name: sha256(path) for path in paths} != hashes or any(
            sha256(source / name) != digest or sha256(snapshot / name) != digest for name, digest in sources.items()):
        raise ValueError("Training input or source changed during fitting")
    model.save_model(str(output / "model.cbm"))
    write_json(output / "report.json", {**metadata, "status": ("FULL_FIT_NOT_OFFICIAL_SCORE"
        if config.data_class == "competition" else "SYNTHETIC_TEST_ONLY"),
        "model_sha256": sha256(output / "model.cbm"), "tree_count": model.tree_count_})
    print("FOLLOWING_BOOST_READY", sha256(output / "model.cbm"), flush=True)


def predict(run: Path, old_run: Path, baseline_path: Path, v10_baseline_path: Path,
        ranking: Path, template_path: Path, output: Path, permission: str) -> None:
    if output.exists() or output.with_suffix(".manifest.json").exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    report = read_object(run / "report.json")
    config = FollowingBoostConfig.model_validate(report.get("config"))
    expected_status = "FULL_FIT_NOT_OFFICIAL_SCORE" if config.data_class == "competition" else "SYNTHETIC_TEST_ONLY"
    if report.get("status") != expected_status or report.get("replacement_weight") != REPLACEMENT_WEIGHT:
        raise ValueError("A completed fit with the fixed replacement contract is required")
    if report.get("tree_count") != config.trees or report.get("clock_rule_version") != nm_clock_overlay.VERSION:
        raise ValueError("Report tree count or clock-rule version differs")
    source = Path(__file__).resolve().parent
    sources = report.get("source_sha256")
    if not isinstance(sources, dict) or not {Path(__file__).name, "nm_following_features.py", "nm_clock_overlay.py"}.issubset(sources):
        raise ValueError("Training source digests are required")
    snapshot = run / "source-snapshot"
    if set(sources) != {path.name for path in snapshot.glob("*.py")}:
        raise ValueError("Training source digests and snapshot coverage differ")
    for name, digest in sources.items():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".py") or not (source / name).is_file() or sha256(source / name) != digest or sha256(snapshot / name) != digest:
            raise ValueError("Training and inference source digests differ")
    paths = [run / "report.json", run / "model.cbm", old_run / "report.json", old_run / "model.cbm",
        baseline_path, baseline_path.with_suffix(".manifest.json"), v10_baseline_path,
        v10_baseline_path.with_suffix(".manifest.json"), ranking, template_path,
        *[source / name for name in sources], *[snapshot / name for name in sources]]
    before = {path: sha256(path) for path in paths}
    ranking_hash, template_hash = before[ranking], before[template_path]
    template = pd.read_parquet(template_path)
    baseline_frame, v11 = checked_baseline(baseline_path, V11_VERSION, ranking_hash, template_hash, template, config.data_class)
    v10_frame, v10 = checked_baseline(v10_baseline_path, V10_VERSION, ranking_hash, template_hash, template, config.data_class)
    if v11.get("baseline_sha256") != before[v10_baseline_path] or v11.get("baseline_model_version") != V10_VERSION or v11.get("baseline_manifest_sha256") != before[v10_baseline_path.with_suffix(".manifest.json")]:
        raise ValueError("The v11 baseline is not bound to the supplied v10 artifact and manifest")
    clock_sources = v11.get("source_sha256")
    if v11.get("proxy_raw_clip") != [-604800, 604800] or v11.get("proxy_valid_inclusive_range") != [-7200, 172800] or not isinstance(clock_sources, dict) or clock_sources.get("nm_clock_overlay.py") != sources["nm_clock_overlay.py"]:
        raise ValueError("The baseline clock-rule contract differs")
    if v10.get("replacement_weight") != REPLACEMENT_WEIGHT or v10.get("residual_fit_cap") != RESIDUAL_CAP or v10.get("baseline_model_version") != "prc2026-ordinary-ensemble/9.0.0":
        raise ValueError("The v10 component weight, residual cap or lineage differs")
    old_report = read_object(old_run / "report.json")
    old_config = NeighborBoostConfig.model_validate(old_report.get("config"))
    if old_report.get("status") != expected_status or old_config.data_class != config.data_class:
        raise ValueError("The original component must be a completed fit with the same data class")
    if old_report.get("model_sha256") != v10.get("new_model_sha256") or before[old_run / "model.cbm"] != v10.get("new_model_sha256") or before[old_run / "report.json"] != v10.get("new_report_sha256"):
        raise ValueError("Old model and report digests are not bound through the v11/v10 manifest chain")
    if old_report.get("input_hashes") != report.get("input_hashes"):
        raise ValueError("Old and new components must use the same twelve training inputs")
    dep, x, old_columns = matrix(pd.read_parquet(ranking))
    if len(x.columns) != 183 or len(old_columns) != 153 or report.get("old_features") != old_columns:
        raise ValueError("Feature schema differs from the documented replacement")
    if len(dep) != len(template) or not pd.Index(dep[ID]).is_unique or not pd.Index(dep[ID]).isin(template[ID]).all():
        raise ValueError("Ranking departures and template IDs must match exactly")
    expected_v11 = apply_clock_overlay(dep.set_index(ID), v10_frame.set_index(ID)[TARGET])
    if not np.array_equal(baseline_frame[ID].map(expected_v11), baseline_frame[TARGET]):
        raise ValueError("The v11 baseline differs from the fixed clock overlay of v10")
    scope = specialist_scope(x)
    if any(manifest.get("specialist_rows") != int(scope.sum()) for manifest in [v11, v10]):
        raise ValueError("Baseline specialist row count differs")
    model, digest = checked_boost(run, report, [str(c) for c in x.columns], shared_configuration(config))
    old_model, old_digest = checked_boost(old_run, old_report, old_columns, old_config)
    offset = nm_schedule_fallback_baseline(x)
    new = pd.Series(np.maximum(0, model.predict(x, thread_count=2) + offset), index=x.index)
    old = pd.Series(np.maximum(0, old_model.predict(x.loc[:, old_columns], thread_count=2) + offset), index=x.index)
    baseline = pd.Series(dep[ID].map(baseline_frame.set_index(ID)[TARGET]).to_numpy(), index=x.index)
    values = replace_component(dep, x, baseline, new, old)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    if {path: sha256(path) for path in paths} != before:
        raise ValueError("Source, model, baseline, ranking or template changed during prediction")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    saved = pd.read_parquet(output)
    validate_submission(template, saved)
    if not np.array_equal(saved[TARGET], result[TARGET]):
        raise ValueError("Serialized predictions differ")
    write_json(output.with_suffix(".manifest.json"), {"data_class": config.data_class, "data_permission_ref": permission,
        "model_version": VERSION, "submission_sha256": sha256(output), "ranking_sha256": ranking_hash,
        "template_sha256": template_hash, "baseline_sha256": before[baseline_path], "baseline_model_version": V11_VERSION,
        "baseline_manifest_sha256": before[baseline_path.with_suffix(".manifest.json")],
        "v10_baseline_sha256": before[v10_baseline_path], "v10_manifest_sha256": before[v10_baseline_path.with_suffix(".manifest.json")],
        "new_model_sha256": digest, "old_model_sha256": old_digest, "new_report_sha256": before[run / "report.json"],
        "old_report_sha256": before[old_run / "report.json"], "source_sha256": sources,
        "replacement_weight": REPLACEMENT_WEIGHT, "residual_fit_cap": RESIDUAL_CAP,
        "clock_rule_version": nm_clock_overlay.VERSION,
        "formula": "v11 + 0.375 * (nonnegative CatBoost183 - nonnegative CatBoost153), outside fixed specialist and clock scopes",
        "rows": len(result), "specialist_rows": int(scope.sum()), "clock_rows": int(clock_scope(dep).sum()),
        "status": "PREDICTIONS_NOT_AN_OFFICIAL_SCORE" if config.data_class == "competition" else "SYNTHETIC_TEST_ONLY"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for name in ["data", "output"]:
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--permission-ref", required=True)
    fit.add_argument("--trees", type=int, default=4999)
    fit.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    fit.add_argument("--data-class", choices=["competition", "synthetic"], default="competition")
    pred = commands.add_parser("predict")
    for name in ["run", "old-run", "baseline", "v10-baseline", "ranking", "template", "output"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, FollowingBoostConfig(trees=args.trees, device=args.device,
            data_class=args.data_class, data_permission_ref=args.permission_ref))
    else:
        predict(args.run, args.old_run, args.baseline, args.v10_baseline, args.ranking, args.template,
            args.output, args.permission_ref)


if __name__ == "__main__":
    main()
