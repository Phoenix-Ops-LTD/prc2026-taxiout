# SPDX-License-Identifier: GPL-3.0-only
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from buckets import TEAM_BUCKET, upload
from pipeline import ID, TARGET, sha256


class FakeS3:
    def __init__(self, objects: list[dict[str, Any]]) -> None:
        self.objects = objects
        self.puts: list[dict[str, Any]] = []

    def get_paginator(self, name: str) -> "FakeS3":
        assert name == "list_objects_v2"
        return self

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs == {"Bucket": TEAM_BUCKET}
        return [{"Contents": self.objects[:2]}, {"Contents": self.objects[2:]}]

    def put_object(self, **kwargs: Any) -> dict[str, str]:
        self.puts.append(kwargs)
        assert kwargs["IfNoneMatch"] == "*"
        assert kwargs["Body"].read()
        return {"ETag": "synthetic-etag"}


def files(tmp_path: Path) -> tuple[Path, Path]:
    template = tmp_path / "template.parquet"
    pd.DataFrame({ID: [2, 1], TARGET: [float("nan")] * 2}).to_parquet(template, index=False)
    output = tmp_path / "zestful-fountain_v6.parquet"
    pd.DataFrame({ID: [2, 1], TARGET: [100., 200.]}).to_parquet(output, index=False)
    output.with_suffix(".manifest.json").write_text(json.dumps({"data_class": "competition",
        "data_permission_ref": "synthetic-upload-test-no-network", "submission_sha256": sha256(output),
        "template_sha256": sha256(template)}))
    return output, template


def objects(n: int) -> list[dict[str, Any]]:
    return [{"Key": f"zestful-fountain_v{i+1}.parquet", "Size": 200,
        "LastModified": datetime.now(timezone.utc) - timedelta(hours=1)} for i in range(n)]


def test_upload_checks_all_pages_and_allows_fifth_submission(tmp_path: Path) -> None:
    output, template = files(tmp_path)
    s3 = FakeS3(objects(4))
    upload(s3, output, template)
    assert len(s3.puts) == 1
    receipt = json.loads(output.with_suffix(".upload.json").read_text())
    assert receipt["previous_submissions_in_24h"] == 4
    assert receipt["status"] == "UPLOADED_SCORE_PENDING"


@pytest.mark.parametrize("case,message", [("quota", "24-hour"), ("version", "version"),
    ("capacity", "one GB"), ("future", "future"), ("naive", "timezone-aware")])
def test_upload_refuses_invalid_remote_state_before_writing(tmp_path: Path, case: str, message: str) -> None:
    output, template = files(tmp_path)
    items = objects(5 if case == "quota" else 1)
    if case == "version":
        items[0]["Key"] = "zestful-fountain_v7.parquet"
    elif case == "capacity":
        items.append({"Key": "result.json", "Size": 1_000_000_000})
    elif case == "future":
        items[0]["LastModified"] = datetime.now(timezone.utc) + timedelta(hours=1)
    elif case == "naive":
        items[0]["LastModified"] = datetime(2026, 1, 1)
    s3 = FakeS3(items)
    with pytest.raises(ValueError, match=message):
        upload(s3, output, template)
    assert not s3.puts
    assert not output.with_suffix(".upload.json").exists()
