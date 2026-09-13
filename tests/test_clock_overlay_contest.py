# SPDX-License-Identifier: GPL-3.0-only
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import clock_overlay_contest as contest
from pipeline import ID, TARGET, sha256, write_json


def artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    clock = pd.Timestamp("2026-01-03T12:00:00Z")
    raw = pd.DataFrame({ID: [1, 2, 3, 4, 5], "PHASE_mvt": ["DEP"] * 4 + ["ARR"],
        "ADEP_mvt": ["LFPG", "LIRF", "LFPG", "LFPG", "OTHER"],
        "ADEP_flt": ["LFPG", "LIRF", "OTHER", "LFPG", "OTHER"],
        "MVT_TIME_UTC_mvt": clock, "IOBT_flt": clock - pd.Timedelta(minutes=20),
        "AOBT_3_flt": clock - pd.Timedelta(minutes=5),
        "LOBT_flt": clock - pd.Timedelta(minutes=130),
        "BLOCK_TIME_UTC_mvt": "FORBIDDEN_DEP_FIELD", TARGET: "FORBIDDEN_DEP_TARGET"})
    raw.loc[1, "IOBT_flt"] = pd.NaT
    raw.loc[3, "LOBT_flt"] = clock - pd.Timedelta(minutes=125)
    baseline_path, ranking, template_path, output = [tmp_path / name for name in
        ["v10.parquet", "ranking.parquet", "template.parquet", "overlay.parquet"]]
    raw.to_parquet(ranking, index=False)
    baseline = pd.DataFrame({ID: [4, 1, 3, 2], TARGET: [400.123456789, 100., 300., 200.]}, index=[9, 8, 7, 6])
    baseline.to_parquet(baseline_path)
    baseline.assign(TAXITIME_SEC_mvt=np.nan).reset_index(drop=True).to_parquet(template_path, index=False)
    write_json(baseline_path.with_suffix(".manifest.json"), {"data_class": "synthetic",
        "model_version": contest.BASELINE_VERSION, "submission_sha256": sha256(baseline_path),
        "ranking_sha256": sha256(ranking), "template_sha256": sha256(template_path),
        "replacement_weight": .375, "residual_fit_cap": 7200,
        "baseline_model_version": "prc2026-ordinary-ensemble/9.0.0", "rows": 4, "specialist_rows": 1})
    return baseline_path, ranking, template_path, output


def test_overlay_round_trip_preserves_template_and_binds_baseline(tmp_path: Path) -> None:
    baseline, ranking, template, output = artifacts(tmp_path)
    contest.predict(baseline, ranking, template, output, "synthetic-test")
    result = pd.read_parquet(output)
    assert result[ID].tolist() == [4, 1, 3, 2]
    np.testing.assert_array_equal(result[TARGET], [400.123456789, 7800., 300., 200.])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["model_version"] == contest.VERSION
    assert manifest["baseline_sha256"] == sha256(baseline)
    assert manifest["baseline_manifest_sha256"] == sha256(baseline.with_suffix(".manifest.json"))
    assert manifest["submission_sha256"] == sha256(output)
    assert manifest["changed_prediction_rows"] == 1 and manifest["specialist_rows"] == 1
    assert manifest["status"] == "SYNTHETIC_TEST_ONLY"
    assert set(manifest["source_sha256"]) == {"clock_overlay_contest.py", "nm_clock_overlay.py", "pipeline.py"}
    with pytest.raises(ValueError, match="fresh"):
        contest.predict(baseline, ranking, template, output, "synthetic-test")


@pytest.mark.parametrize("field,value", [("model_version", "other"), ("data_class", "other"),
    ("submission_sha256", "0" * 64), ("ranking_sha256", "0" * 64), ("template_sha256", "0" * 64),
    ("replacement_weight", .5), ("residual_fit_cap", 3600), ("baseline_model_version", "other"),
    ("rows", 3), ("specialist_rows", 0)])
def test_rejects_inconsistent_baseline_manifest(tmp_path: Path, field: str, value: object) -> None:
    baseline, ranking, template, output = artifacts(tmp_path)
    path = baseline.with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    manifest[field] = value
    write_json(path, manifest)
    with pytest.raises(ValueError):
        contest.predict(baseline, ranking, template, output, "synthetic-test")
    assert not output.exists()


def test_rejects_input_change_during_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline, ranking, template, output = artifacts(tmp_path)
    original = contest.apply_clock_overlay

    def changed(movements: pd.DataFrame, predictions: pd.Series) -> pd.Series:
        values = original(movements, predictions)
        with ranking.open("ab") as stream:
            stream.write(b"changed during operation")
        return values

    monkeypatch.setattr(contest, "apply_clock_overlay", changed)
    with pytest.raises(ValueError, match="changed during prediction"):
        contest.predict(baseline, ranking, template, output, "synthetic-test")
    assert not output.exists()


def test_rejects_missing_authorization_and_manifest_output_collision(tmp_path: Path) -> None:
    baseline, ranking, template, output = artifacts(tmp_path)
    with pytest.raises(ValueError, match="authorization"):
        contest.predict(baseline, ranking, template, output, " ")
    output.with_suffix(".manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh"):
        contest.predict(baseline, ranking, template, output, "synthetic-test")
    assert not output.exists()
