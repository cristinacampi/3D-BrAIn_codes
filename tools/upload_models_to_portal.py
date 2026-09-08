#!/usr/bin/env python3
"""Bulk-upload trained catalogue models to the 3D-BrAIn portal R2 storage."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import ssl
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import certifi
except ImportError:
    certifi = None

DEFAULT_PORTAL = "https://brain-open-research.cristina-campi.chatgpt.site"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = REPOSITORY_ROOT / "outputs" / "analisi_dati_Mark"
DEFAULT_TOKEN_FILE = Path(__file__).resolve().with_name(".env.portal-upload")


def read_secrets(path: Path | None) -> tuple[str, str]:
    values = dict(os.environ)
    if path:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values.setdefault(key.strip(), value.strip().strip("'\""))
    token = values.get("MODEL_UPLOAD_TOKEN", "")
    bypass = values.get("SITES_BYPASS_TOKEN", "")
    if not token:
        raise SystemExit("MODEL_UPLOAD_TOKEN is missing. Use --token-file or the environment variable.")
    return token, bypass


def scope_from_catalog(catalog: Path) -> str | None:
    parts = catalog.parts
    experiment = next((part for part in parts if part.startswith("experiment_")), None)
    campaign = catalog.parent.parent.name
    if not experiment:
        return None
    target = "culture-phase" if "culture_phase" in campaign else "div" if "_div_" in campaign else None
    return f"experiment-{experiment.removeprefix('experiment_')}-{target}" if target else None


def discover(root: Path) -> list[tuple[str, Path]]:
    uploads: list[tuple[str, Path]] = []
    for catalog in sorted(root.rglob("final_models/portal_model_catalog.json")):
        scope = scope_from_catalog(catalog)
        if not scope:
            continue
        records = json.loads(catalog.read_text(encoding="utf-8"))
        for record in records:
            model_id = record.get("id")
            if not isinstance(model_id, str):
                continue
            candidates = [catalog.parent / f"{model_id}.pt", catalog.parent / f"{model_id}.joblib"]
            artifact = next((candidate for candidate in candidates if candidate.is_file()), None)
            if artifact:
                uploads.append((f"{scope}-{model_id}", artifact))
    return uploads


def multipart(model_id: str, artifact: Path, overwrite: bool) -> tuple[bytes, str]:
    boundary = f"----3dbrain-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in (("modelId", model_id), ("overwrite", str(overwrite).lower())):
        chunks.extend([f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()])
    content_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
    chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{artifact.name}\"\r\nContent-Type: {content_type}\r\n\r\n".encode())
    chunks.append(artifact.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def upload(portal: str, token: str, bypass: str, model_id: str, artifact: Path, overwrite: bool) -> dict:
    body, boundary = multipart(model_id, artifact, overwrite)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": f"multipart/form-data; boundary={boundary}"}
    if bypass:
        headers["OAI-Sites-Authorization"] = f"Bearer {bypass}"
    request = Request(f"{portal.rstrip('/')}/api/models/upload", data=body, headers=headers, method="POST")
    try:
        context = ssl.create_default_context(cafile=certifi.where()) if certifi else ssl.create_default_context()
        with urlopen(request, timeout=180, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(str(error.reason)) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--portal-url", default=DEFAULT_PORTAL)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Upload only the first N matched artifacts (useful for testing).")
    args = parser.parse_args()

    uploads = discover(args.root.resolve())
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1.")
        uploads = uploads[: args.limit]
    if not uploads:
        print("No model artifacts matched the portal catalogues.", file=sys.stderr)
        return 1
    print(f"Found {len(uploads)} model artifacts.")
    if args.dry_run:
        for model_id, artifact in uploads:
            print(f"DRY RUN  {model_id}  {artifact.name}  {artifact.stat().st_size / 1024 / 1024:.2f} MB")
        return 0

    token, bypass = read_secrets(args.token_file)
    completed = failed = skipped = 0
    for index, (model_id, artifact) in enumerate(uploads, start=1):
        print(f"[{index}/{len(uploads)}] {model_id} ({artifact.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
        try:
            result = upload(args.portal_url, token, bypass, model_id, artifact, args.overwrite)
            print(f"  uploaded: {result['r2Url']}  sha256={result['sha256']}", flush=True)
            completed += 1
        except RuntimeError as error:
            if "already exists" in str(error):
                print("  skipped: already present", flush=True)
                skipped += 1
            else:
                print(f"  ERROR: {error}", file=sys.stderr, flush=True)
                failed += 1
    print(f"Finished: {completed} uploaded, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
