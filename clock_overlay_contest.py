# SPDX-License-Identifier: GPL-3.0-only
"""Apply the fixed NM-clock diagnostic to a verified v10 prediction artifact.

Retrospective competition research. This command does not fit a model, upload
predictions, select a rule, or establish an official improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import nm_clock_overlay
import pipeline
from nm_clock_overlay import apply_clock_overlay as apply_clock_overlay
from pipeline import ID, TARGET, sha256, validate_submission, write_json

VERSION = "prc2026-nm-clock-overlay/11.0.0"
BASELINE_VERSION = "prc2026-neighbor-boost-replacement/10.0.0"
MOVEMENT_COLUMNS = [ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "MVT_TIME_UTC_mvt",
    "IOBT_flt", "AOBT_3_flt", "LOBT_flt"]


def read_object(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("A JSON object is required")
    return value


def predict(baseline_path: Path, ranking: Path, template_path: Path,
        output: Path, permission: str) -> None:
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    baseline_manifest = baseline_path.with_suffix(".manifest.json")
    sources = [Path(__file__).resolve(), Path(nm_clock_overlay.__file__).resolve(), Path(pipeline.__file__).resolve()]
    paths = [baseline_path, baseline_manifest, ranking, template_path, *sources]
    before = {path: sha256(path) for path in paths}
    manifest = read_object(baseline_manifest)
    if manifest.get("model_version") != BASELINE_VERSION or manifest.get("data_class") not in {"competition", "synthetic"}:
        raise ValueError("The exact v10 baseline version and explicit data class are required")
    if manifest.get("submission_sha256") != before[baseline_path]:
        raise ValueError("Baseline digest mismatch")
    if manifest.get("ranking_sha256") != before[ranking] or manifest.get("template_sha256") != before[template_path]:
        raise ValueError("Baseline ranking or template digest mismatch")
    if manifest.get("replacement_weight") != .375 or manifest.get("residual_fit_cap") != 7200:
        raise ValueError("The baseline component weight or residual fitting cap differs")
    if manifest.get("baseline_model_version") != "prc2026-ordinary-ensemble/9.0.0":
        raise ValueError("The supplied v10 must retain its documented v9 lineage")
    template, baseline = pd.read_parquet(template_path), pd.read_parquet(baseline_path)
    validate_submission(template, baseline)
    if manifest.get("rows") != len(baseline):
        raise ValueError("Baseline manifest row count differs")
    # Never load DEP target/block fields from the ranking data.
    raw = pd.read_parquet(ranking, columns=MOVEMENT_COLUMNS)
    dep = raw.loc[raw["PHASE_mvt"].eq("DEP")].set_index(ID)
    values = baseline.set_index(ID)[TARGET]
    result_values = apply_clock_overlay(dep, values)
    scope = dep["ADEP_mvt"].eq("LIRF") & dep["IOBT_flt"].isna()
    if manifest.get("specialist_rows") != int(scope.sum()):
        raise ValueError("The baseline specialist row count differs")
    specialist_ids = dep.index[scope]
    if not np.array_equal(result_values.loc[specialist_ids], values.loc[specialist_ids]):
        raise ValueError("Specialist predictions must be preserved exactly")
    result = template.copy()
    result[TARGET] = result[ID].map(result_values)
    validate_submission(template, result)
    if {path: sha256(path) for path in paths} != before:
        raise ValueError("Source, baseline, ranking or template changed during prediction")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    saved = pd.read_parquet(output)
    validate_submission(template, saved)
    if not np.array_equal(saved[TARGET], result[TARGET]):
        raise ValueError("Serialized predictions differ")
    write_json(manifest_path, {"data_class": manifest["data_class"], "data_permission_ref": permission,
        "model_version": VERSION, "submission_sha256": sha256(output),
        "ranking_sha256": before[ranking], "template_sha256": before[template_path],
        "baseline_sha256": before[baseline_path], "baseline_manifest_sha256": before[baseline_manifest],
        "baseline_model_version": BASELINE_VERSION, "source_sha256": {path.name: before[path] for path in sources},
        "rule": "ordinary, NM origin match, both original clock proxies valid, AOBT minus LOBT strictly above 7200 seconds",
        "proxy_raw_clip": [-604800, 604800], "proxy_valid_inclusive_range": [-7200, 172800],
        "replacement": "entire prediction replaced by nonnegative movement-minus-LOBT proxy on the fixed scope",
        "rows": len(result), "specialist_rows": int(scope.sum()),
        "changed_prediction_rows": int(np.count_nonzero(result[TARGET].to_numpy() != baseline[TARGET].to_numpy())),
        "status": "PREDICTIONS_NOT_AN_OFFICIAL_SCORE" if manifest["data_class"] == "competition" else "SYNTHETIC_TEST_ONLY"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["baseline", "ranking", "template", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    predict(args.baseline, args.ranking, args.template, args.output, args.permission_ref)


if __name__ == "__main__":
    main()
