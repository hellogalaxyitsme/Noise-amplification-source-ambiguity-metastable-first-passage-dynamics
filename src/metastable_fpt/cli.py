"""Command-line interface.

    python -m metastable_fpt list
    python -m metastable_fpt download-data [--data-dir data/markers]
    python -m metastable_fpt run <analysis | all> [--quick] [--out results] [--data-dir data/markers] [--workers N]

Each run writes ``<out>/<analysis>/results.json`` together with the parameters
used (``parameters.json``), the software environment (``environment.json``) and
the wall-clock time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .analyses import ANALYSES
from .data_download import download_markers, verify_markers
from .results_io import environment, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "markers"
MANIFEST = REPO_ROOT / "data" / "marker_checksums.sha256"


def _run(name: str, args) -> None:
    module = ANALYSES[name]
    params = module.parameters(quick=args.quick)
    out = Path(args.out) / name
    print(f"[{name}] {module.DESCRIPTION}{' (quick settings)' if args.quick else ''}", flush=True)
    if name == "human_waiting_times" and MANIFEST.exists():
        problems = verify_markers(Path(args.data_dir), MANIFEST)
        if problems:
            raise SystemExit(f"marker files do not match {MANIFEST.name}: {problems}; run 'download-data' first")
    t0 = time.time()
    results = module.run(params, data_dir=args.data_dir, workers=args.workers)
    elapsed = time.time() - t0
    write_json(out / "parameters.json", {"analysis": name, "quick": args.quick, "parameters": params})
    write_json(out / "environment.json", {**environment(), "elapsed_s": elapsed})
    write_json(out / "results.json", results)
    print(f"[{name}] done in {elapsed:.0f} s -> {out / 'results.json'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m metastable_fpt")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the analyses")
    dl = sub.add_parser("download-data", help="download the header and marker files from OSF")
    dl.add_argument("--data-dir", default=str(DEFAULT_DATA))
    run = sub.add_parser("run", help="run one analysis or all of them")
    run.add_argument("analysis", choices=sorted(ANALYSES) + ["all"])
    run.add_argument("--quick", action="store_true", help="reduced settings for a fast check")
    run.add_argument("--out", default="results")
    run.add_argument("--data-dir", default=str(DEFAULT_DATA))
    run.add_argument("--workers", type=int, default=None, help="processes for the accumulator fits")
    args = ap.parse_args(argv)

    if args.command == "list":
        for name, module in ANALYSES.items():
            print(f"{name:34s} {module.DESCRIPTION}")
        return 0
    if args.command == "download-data":
        files = download_markers(Path(args.data_dir), MANIFEST if MANIFEST.exists() else None)
        print(f"downloaded {len(files)} files to {args.data_dir}")
        return 0
    names = list(ANALYSES) if args.analysis == "all" else [args.analysis]
    for name in names:
        _run(name, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
