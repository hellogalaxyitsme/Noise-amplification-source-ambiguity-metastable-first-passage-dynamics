"""Download the BrainVision header and marker files of the Libet-clock dataset from OSF.

Source: Open Science Framework project 3cvyk (https://doi.org/10.17605/OSF.IO/3CVYK),
released by Jeay-Bizot, Chishti, Maoz and Schurger with their PLOS Biology article
(2026). Only ``P*_RespRP.vhdr`` and ``P*_RespRP.vmrk`` are downloaded (about
0.4 MB in total); the EEG signal files are not needed. Each downloaded file is
checked against the SHA-256 manifest ``data/marker_checksums.sha256`` when that
manifest is present.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

OSF_FILES = "https://api.osf.io/v2/nodes/3cvyk/files/osfstorage/?page[size]=100"
WANTED = re.compile(r"^P\d+_RespRP\.(vhdr|vmrk)$")
USER_AGENT = "metastable-fpt/1.0"


def _get(url: str, *, attempts: int = 4, timeout: int = 120) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed public host
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts: {last}")


def list_marker_files() -> list[dict]:
    """Name, size and download link of every header and marker file in the OSF project."""

    url: str | None = OSF_FILES
    files: list[dict] = []
    while url:
        data = json.loads(_get(url))
        for item in data.get("data", []):
            attrs = item["attributes"]
            if attrs.get("kind") == "file" and WANTED.match(attrs.get("name", "")):
                files.append({"name": attrs["name"], "size": attrs.get("size"),
                              "download": (item.get("links") or {}).get("download")})
        url = (data.get("links") or {}).get("next")
    return sorted(files, key=lambda f: f["name"])


def read_manifest(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split()
                out[name] = digest
    return out


def download_markers(target: Path, manifest: Path | None = None) -> list[Path]:
    """Download all header and marker files into ``target``; verify sizes and checksums."""

    target.mkdir(parents=True, exist_ok=True)
    expected = read_manifest(manifest) if manifest is not None else {}
    files = list_marker_files()
    if not files:
        raise RuntimeError("no P*_RespRP.vhdr/.vmrk files found in OSF project 3cvyk")
    written = []
    for f in files:
        blob = _get(f["download"])
        if f["size"] is not None and len(blob) != int(f["size"]):
            raise RuntimeError(f"{f['name']}: size {len(blob)} differs from the listed {f['size']}")
        digest = hashlib.sha256(blob).hexdigest()
        if f["name"] in expected and expected[f["name"]] != digest:
            raise RuntimeError(f"{f['name']}: SHA-256 {digest} differs from the manifest")
        path = target / f["name"]
        path.write_bytes(blob)
        written.append(path)
    missing = sorted(set(expected) - {p.name for p in written})
    if missing:
        raise RuntimeError(f"files listed in the manifest were not found on OSF: {missing}")
    return written


def verify_markers(directory: Path, manifest: Path) -> list[str]:
    """Names whose SHA-256 differs from the manifest or that are missing (empty if all match)."""

    problems = []
    for name, digest in read_manifest(manifest).items():
        path = directory / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"{name}: checksum mismatch")
    return problems
