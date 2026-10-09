# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import hashlib
import zipfile

import pytest

from forecast_delivery import export_source


def test_export_excludes_private_runs_weights_and_data(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ["LICENSE", "pyproject.toml", "requirements.lock.txt", ".gitignore", "config.example.json", "README.md", "model.py"]:
        (source / name).write_text("public source\n", encoding="utf-8")
    private = source / "runs"
    private.mkdir()
    (private / "training.parquet").write_bytes(b"restricted")
    (source / "weights.cbm").write_bytes(b"restricted")
    (source / "credentials.json").write_text("private", encoding="utf-8")
    output = tmp_path / "public.zip"
    report = export_source(source, output)
    assert report["restricted_artifacts"] == "EXCLUDED"
    with zipfile.ZipFile(output) as archive:
        assert "model.py" in archive.namelist()
        assert not any("runs" in n or "credentials" in n or "weights" in n for n in archive.namelist())
        assert report["files"] == {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}
    with pytest.raises(ValueError, match="new"):
        export_source(source, output)


def test_export_stops_before_writing_source_with_credentials(tmp_path: Path) -> None:
    for name in ["LICENSE", "pyproject.toml", "requirements.lock.txt", ".gitignore", "config.example.json"]:
        (tmp_path / name).write_text("public\n", encoding="utf-8")
    (tmp_path / "unsafe.py").write_text("password" + "=" + "abcdefghijklmnop", encoding="utf-8")
    output = tmp_path / "refused.zip"
    with pytest.raises(ValueError, match="credentials"):
        export_source(tmp_path, output)
    assert not output.exists()
