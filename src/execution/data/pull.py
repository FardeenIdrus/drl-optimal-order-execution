"""Stage 1 - pull raw L2 files from the source into scratch (outside the repo).

Reads the Stage 0 manifest and downloads every present hourly file, mirroring the
key layout locally as ``<dest>/<date>/<hour>/<COIN>.lz4``. Resumable and
idempotent: a file already present locally with the manifest's size is skipped, so
re-running only fetches what is missing. Parallel via the AWS CLI (requester-pays).

Usage:
    PYTHONPATH=src python -m execution.data.pull \
        --manifest <scratch>/manifest/BTC_..._manifest.csv \
        --dest <scratch>/raw_l2/BTC --workers 16
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from execution.data.sources.hyperliquid_l2 import AWS_BIN, BUCKET

_print_lock = threading.Lock()


def _local_path(dest: Path, date: str, hour: int, coin: str) -> Path:
    return dest / date / str(hour) / f"{coin}.lz4"


def _download_one(key: str, size: int, target: Path) -> str:
    """Download one object unless already present with the expected size.

    Returns 'skip', 'ok', or 'err:<msg>'.
    """
    if target.exists() and target.stat().st_size == size:
        return "skip"
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [AWS_BIN, "s3", "cp", f"s3://{BUCKET}/{key}", str(target),
           "--request-payer", "requester", "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return f"err:{proc.stderr.strip()[:200]}"
    if not (target.exists() and target.stat().st_size == size):
        return "err:size-mismatch-after-download"
    return "ok"


def pull(manifest_csv: str, dest: str, coin: str = "BTC", workers: int = 16) -> dict:
    dest_p = Path(dest)
    with open(manifest_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["present"] == "True"]

    tasks = [(r["key"], int(r["size_bytes"]),
              _local_path(dest_p, r["date"], int(r["hour"]), coin)) for r in rows]
    total = len(tasks)
    counts = {"ok": 0, "skip": 0, "err": 0}
    errors: List[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, k, s, t): (k, t) for k, s, t in tasks}
        for fut in as_completed(futs):
            key, _ = futs[fut]
            res = fut.result()
            done += 1
            if res == "ok":
                counts["ok"] += 1
            elif res == "skip":
                counts["skip"] += 1
            else:
                counts["err"] += 1
                errors.append(f"{key}: {res}")
            if done % 500 == 0 or done == total:
                with _print_lock:
                    print(f"  [{done}/{total}] ok={counts['ok']} "
                          f"skip={counts['skip']} err={counts['err']}", flush=True)

    print(f"DONE: {counts['ok']} downloaded, {counts['skip']} already present, "
          f"{counts['err']} errors (of {total}).")
    if errors:
        print("first errors:")
        for e in errors[:10]:
            print("  " + e)
    return {"total": total, **counts, "errors": errors}


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1 - pull raw L2 files")
    p.add_argument("--manifest", required=True)
    p.add_argument("--dest", required=True, help="scratch dir, OUTSIDE the repo")
    p.add_argument("--coin", default="BTC")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    pull(args.manifest, args.dest, args.coin, args.workers)


if __name__ == "__main__":
    main()
