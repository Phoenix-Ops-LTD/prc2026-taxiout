# SPDX-License-Identifier: GPL-3.0-only
"""Audit all evaluation columns; verify exact exported prediction/source artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from forecast_features import clean_departures, schedule_features
from forecast_model import FittedForecast
from forecast_novel_guard import load_model
from pipeline import ID, TARGET, sha256, validate_submission, write_json


def evaluation_audit(ranking: Path, final: Path, output: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, path in [("ranking", ranking), ("final", final)]:
        table = pq.read_table(path)
        reports[name] = {"sha256": sha256(path), "rows": table.num_rows,
                         "missing_all_columns": {n: table.column(n).null_count for n in table.column_names},
                         "schema": [(f.name, str(f.type)) for f in table.schema]}
        del table
    earlier = pd.read_parquet(ranking)
    later = pd.read_parquet(final)
    common = sorted(set(earlier[ID]) & set(later[ID]))
    a, b = earlier.set_index(ID).loc[common], later.set_index(ID).loc[common]
    differences: dict[str, int] = {}
    for column in set(a.columns) & set(b.columns):
        same = a[column].eq(b[column]) | (a[column].isna() & b[column].isna())
        differences[str(column)] = int((~same).fillna(True).sum())
    reports["ranking_final_shared_movements"] = len(common)
    reports["shared_column_differences"] = differences
    write_json(output, reports)
    return reports


def verify(run: Path, ranking: Path, template: Path, submission: Path, output: Path) -> dict[str, Any]:
    """Fresh inference on all rows, then match the serialized, template-ordered file."""
    manifest = json.loads((run / "model" / "model-manifest.json").read_text(encoding="utf-8"))
    raw = pd.read_parquet(ranking)
    data = clean_departures(raw)
    x = schedule_features(data, raw)
    prediction = np.zeros(len(data))
    components: dict[str, Any] = {}
    for name, weight in manifest["weights"].items():
        model = load_model(run / "model" / name)
        p = model.predict(x)
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Invalid component predictions")
        prediction += weight * p
        components[name] = {"weight": weight, "metadata_sha256": sha256(run / "model" / name / "metadata.json")}
    expected = pd.read_parquet(template)
    expected[TARGET] = expected[ID].map(pd.Series(prediction, index=data[ID])).astype(float)
    actual = pd.read_parquet(submission)
    validate_submission(pd.read_parquet(template), actual)
    if not expected.equals(actual):
        raise ValueError("Fresh native inference differs from serialized predictions")
    provenance = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    for name, digest in provenance["source_hashes"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError("Training source changed after its recorded hash")
    receipt = {"status": "ALL_ROWS_EXACT_FRESH_NATIVE_INFERENCE_PASS", "rows": len(actual),
               "submission_sha256": sha256(submission), "template_sha256": sha256(template),
               "ranking_sha256": sha256(ranking), "components": components,
               "source_bindings_unchanged": True}
    write_json(output, receipt)
    return receipt


def export_source(source: Path, output: Path, inference_only: bool = False) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Source archive must be new")
    paths = sorted([*source.glob("*.py"), *source.glob("*.md"), *source.glob("tests/*.py"),
                    *source.glob("results/*.md"), *source.glob("results/*.json"), *source.glob(".github/workflows/*.yml"),
                    source / "LICENSE", source / "pyproject.toml", source / "requirements.lock.txt",
                    source / ".gitignore", source / "config.example.json"])
    if (source / "leaderboard.json").exists():
        paths.append(source / "leaderboard.json")
    if (source / "dashboard.html").exists():
        paths.append(source / "dashboard.html")
    if inference_only:
        allowed = {"taxiout.py", "forecast_model.py", "forecast_guard.py", "forecast_novel_guard.py", "forecast_features.py", "pipeline.py", "LICENSE",
                   "requirements.lock.txt", "DATA_USE.md", "GENERALIZATION.md", "INFERENCE.md"}
        paths = [p for p in paths if p.name in allowed]
    pattern = re.compile(r"(api[_-]?key|client[_-]?secret|access[_-]?token|private[_-]?key|password|secret[_-]?key)\b[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9+/_.-]{12,}", re.I)
    # Snapshot changing live leaderboard files once; manifests describe ZIP bytes.
    payloads = {p: p.read_bytes() for p in paths}
    findings: list[str] = []
    for path in paths:
        text = payloads[path].decode("utf-8")
        private_key_header = "-----BEGIN " + "PRIVATE KEY-----"
        if pattern.search(text) or private_key_header in text:
            findings.append(str(path.relative_to(source)))
    if findings:
        raise ValueError("Potential source credentials found in: " + ", ".join(findings))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            name = "README.md" if inference_only and path.name == "INFERENCE.md" else path.relative_to(source).as_posix()
            archive.writestr(name, payloads[path])
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity failure")
        for name in archive.namelist():
            if name.startswith(("runs/", ".venv/", "../")) or name.endswith((".parquet", ".cbm", ".joblib")):
                raise ValueError("Restricted artifacts in export")
    receipt = {"status": "LOCAL_SOURCE_ARCHIVE_READY_NOT_PUBLISHED", "archive_sha256": sha256(output),
               "files": {("README.md" if inference_only and p.name == "INFERENCE.md" else p.relative_to(source).as_posix()): hashlib.sha256(payloads[p]).hexdigest() for p in paths},
               "secret_assignment_scan": "PASS", "restricted_artifacts": "EXCLUDED",
               "licence": "GPL-3.0-only", "automatically_published": False,
               "profile": "PORTABLE_INFERENCE_SOURCE_WITHOUT_WEIGHTS" if inference_only else "COMPLETE_RESEARCH_SOURCE"}
    write_json(output.with_suffix(".manifest.json"), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    for name in ["ranking", "final", "output"]:
        audit.add_argument("--" + name, type=Path, required=True)
    checker = commands.add_parser("verify")
    for name in ["run", "ranking", "template", "submission", "output"]:
        checker.add_argument("--" + name, type=Path, required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--source", type=Path, default=Path(__file__).parent)
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        result = evaluation_audit(args.ranking, args.final, args.output)
    elif args.command == "verify":
        result = verify(args.run, args.ranking, args.template, args.submission, args.output)
    else:
        result = export_source(args.source, args.output, args.inference_only)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
