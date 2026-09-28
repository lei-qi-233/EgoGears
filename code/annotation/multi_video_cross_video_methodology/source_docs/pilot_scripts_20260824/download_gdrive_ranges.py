#!/usr/bin/env python3
"""Resume public Google Drive files through validated HTTP range requests.

This is a fallback for large shared files when the normal download endpoint
intermittently returns a quota HTML page. Chunks are accepted only when both
HTTP Content-Range and byte length match the request.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CHUNK_SIZE = 16 * 1024 * 1024
CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def drive_id(url: str) -> str:
    values = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("id")
    if not values:
        raise ValueError(f"No id in URL: {url}")
    return values[0]


def request_range(file_id: str, start: int, end: int, attempts: int = 60) -> tuple[bytes, int]:
    url = (
        "https://drive.usercontent.google.com/download?"
        + urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"})
    )
    expected = end - start + 1
    last_error = "unknown"
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
                match = CONTENT_RANGE_RE.fullmatch(response.headers.get("Content-Range", ""))
                if match:
                    got_start, got_end, total = map(int, match.groups())
                    if got_start == start and got_end == end and len(data) == expected:
                        return data, total
                last_error = (
                    f"status={response.status} bytes={len(data)} "
                    f"content_range={response.headers.get('Content-Range')}"
                )
        except (OSError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        delay = min(2.0, 0.25 * (attempt + 1)) + random.random()
        time.sleep(delay)
    raise RuntimeError(f"range {start}-{end} failed after {attempts} attempts: {last_error}")


def get_total(file_id: str) -> int:
    _, total = request_range(file_id, 0, 0)
    return total


def save_state(path: Path, completed: set[int]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sorted(completed)) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download_one(root: Path, relative_path: str, file_id: str) -> str:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    total = get_total(file_id)
    if target.exists() and target.stat().st_size == total:
        return f"skip complete: {relative_path}"

    part = target.with_suffix(target.suffix + ".part")
    state = target.with_suffix(target.suffix + ".part.state.json")
    if state.exists():
        completed = set(json.loads(state.read_text(encoding="utf-8")))
    else:
        completed = set()
        with part.open("wb") as handle:
            handle.truncate(total)

    chunk_count = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    with part.open("r+b", buffering=0) as handle:
        for index in range(chunk_count):
            if index in completed:
                continue
            start = index * CHUNK_SIZE
            end = min(total - 1, start + CHUNK_SIZE - 1)
            data, reported_total = request_range(file_id, start, end)
            if reported_total != total:
                raise RuntimeError(
                    f"size changed for {relative_path}: {total} -> {reported_total}"
                )
            handle.seek(start)
            handle.write(data)
            completed.add(index)
            save_state(state, completed)
            print(f"{relative_path}: {len(completed)}/{chunk_count} chunks", flush=True)
        os.fsync(handle.fileno())

    if part.stat().st_size != total or len(completed) != chunk_count:
        raise RuntimeError(f"incomplete file: {relative_path}")
    os.replace(part, target)
    state.unlink(missing_ok=True)
    return f"complete: {relative_path} ({total} bytes)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("listing", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    entries = json.loads(args.listing.read_text(encoding="utf-8"))
    jobs = [(entry["path"], drive_id(entry["url"])) for entry in entries]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, args.root, relative_path, file_id): relative_path
            for relative_path, file_id in jobs
        }
        for future in as_completed(futures):
            relative_path = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append(relative_path)
                print(f"FAILED {relative_path}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} downloads failed")


if __name__ == "__main__":
    main()
