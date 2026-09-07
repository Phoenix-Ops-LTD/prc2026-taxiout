# SPDX-License-Identifier: GPL-3.0-only
"""Competition S3 access. REST OAuth credentials are not interchangeable with S3 keys."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config
from pydantic import AliasChoices, BaseModel, Field, SecretStr

from pipeline import sha256, validate_submission, write_json

ENDPOINT = "https://s3.opensky-network.org"
TEAM_BUCKET = "prc-2026-zestful-fountain"


class BucketCredentials(BaseModel):
    access_key: SecretStr = Field(validation_alias=AliasChoices("accessKey", "access_key", "bucket_access_key"))
    secret_key: SecretStr = Field(validation_alias=AliasChoices("secretKey", "secret_key", "bucket_access_secret"))


def client(path: Path) -> Any:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "clientId" in raw or "clientSecret" in raw:
        raise ValueError("This file contains REST OAuth credentials. Generate separate MinIO access keys through OpenSky SSO.")
    # Never print validation errors, which may contain input values.
    try:
        keys = BucketCredentials.model_validate(raw)
    except Exception:
        raise ValueError("MinIO credentials need accessKey and secretKey fields") from None
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=keys.access_key.get_secret_value(), aws_secret_access_key=keys.secret_key.get_secret_value(), region_name="us-east-1", config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=60, retries={"max_attempts": 2}))


def upload(s3: Any, path: Path, template: Path) -> None:
    if not re.fullmatch(r"zestful-fountain_v[1-9][0-9]*\.parquet", path.name):
        raise ValueError("Use zestful-fountain_vN.parquet with an increasing positive integer")
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if manifest["data_class"] != "competition" or not manifest.get("data_permission_ref"):
        raise ValueError("Synthetic or unlicensed runs cannot be uploaded")
    if manifest["submission_sha256"] != sha256(path) or manifest["template_sha256"] != sha256(template):
        raise ValueError("Submission or template digest changed")
    validate_submission(pd.read_parquet(template), pd.read_parquet(path))
    # Conditional create prevents accidental replacement of a previous submission.
    with path.open("rb") as body:
        response = s3.put_object(Bucket=TEAM_BUCKET, Key=path.name, Body=body, ContentType="application/vnd.apache.parquet", IfNoneMatch="*")
    write_json(path.with_suffix(".upload.json"), {"bucket": TEAM_BUCKET, "key": path.name, "submission_sha256": manifest["submission_sha256"], "etag": response.get("ETag"), "status": "UPLOADED_SCORE_PENDING"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-buckets")
    listing = commands.add_parser("list")
    listing.add_argument("--bucket", required=True)
    get = commands.add_parser("download")
    get.add_argument("--bucket", required=True)
    get.add_argument("--key", required=True)
    get.add_argument("--output", type=Path, required=True)
    put = commands.add_parser("upload")
    put.add_argument("--submission", type=Path, required=True)
    put.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    try:
        s3 = client(args.credentials)
        if args.command == "list-buckets":
            print("\n".join(b["Name"] for b in s3.list_buckets()["Buckets"]))
        elif args.command == "list":
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket):
                for item in page.get("Contents", []):
                    print(item["Key"], item["Size"])
        elif args.command == "download":
            if args.output.exists():
                raise ValueError("Use a fresh output path")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(args.bucket, args.key, str(args.output))
            write_json(args.output.with_suffix(".download.json"), {"endpoint": ENDPOINT, "bucket": args.bucket, "key": args.key, "sha256": sha256(args.output)})
        else:
            upload(s3, args.submission, args.template)
            print("Uploaded to team bucket; score receipt still required")
    except Exception as error:
        # Provider exceptions may expose request details: show only safe class names.
        print(f"Bucket operation failed ({type(error).__name__}). Check key type, access, file manifest and naming; no credentials logged.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
