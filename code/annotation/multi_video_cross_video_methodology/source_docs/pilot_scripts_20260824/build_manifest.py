#!/usr/bin/env python3
"""Build a metadata manifest for the CVHCI running video collection."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import av


NAME_RE = re.compile(
    r"^(?P<participant>P\d{2})_"
    r"(?P<activity>FastWalk|SlowWalk|Running)_"
    r"(?P<lighting>Day|Night)_"
    r"(?:(?:Traj(?P<trajectory>\d{2}))(?:_Trial(?P<trial_after_traj>\d{2}))?"
    r"|(?:Trial(?P<trial>\d{2})))$"
)


def probe(path: Path) -> dict:
    with av.open(str(path)) as container:
        video = next((stream for stream in container.streams if stream.type == "video"), None)
        audio = next((stream for stream in container.streams if stream.type == "audio"), None)
        metadata_keys = {key.lower() for key in container.metadata}
        return {
            "duration_s": round(float(container.duration / av.time_base), 6)
            if container.duration
            else None,
            "size_bytes": path.stat().st_size,
            "bit_rate": container.bit_rate,
            "video_codec": video.codec_context.name if video else None,
            "width": video.codec_context.width if video else None,
            "height": video.codec_context.height if video else None,
            "fps": round(float(video.average_rate), 6)
            if video and video.average_rate
            else None,
            "video_frames": video.frames if video else None,
            "has_audio": audio is not None,
            "audio_codec": audio.codec_context.name if audio else None,
            "audio_sample_rate": audio.codec_context.sample_rate if audio else None,
            "audio_channels": audio.codec_context.channels if audio else None,
            # Record privacy-risk flags without copying sensitive values such
            # as coordinates or capture timestamps into the derived manifest.
            "has_location_metadata": any("location" in key for key in metadata_keys),
            "has_creation_time_metadata": any("creation" in key for key in metadata_keys),
            "stream_types": [stream.type for stream in container.streams],
        }


def parse_labels(path: Path) -> dict:
    match = NAME_RE.fullmatch(path.stem)
    if not match:
        return {"parse_ok": False}
    labels = match.groupdict()
    trajectory = labels.pop("trajectory")
    trial = labels.pop("trial")
    trial_after_traj = labels.pop("trial_after_traj")
    trial = trial or trial_after_traj
    # The source naming is inconsistent: P01 uses Traj, P02 uses Trial, and
    # P03 uses both. Preserve those semantics and expose only a neutral
    # sequence index for balancing; do not assume that Trial means route.
    sequence_index = int(trajectory or trial)
    return {
        "parse_ok": True,
        **labels,
        "trajectory_index": int(trajectory) if trajectory else None,
        "trial_index": int(trial) if trial else None,
        "sequence_index": sequence_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.root.rglob("*.mp4"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for path in paths:
        relative_path = path.relative_to(args.root).as_posix()
        row = {"id": path.stem, "relative_path": relative_path, **parse_labels(path)}
        try:
            row.update(probe(path))
        except Exception as exc:  # Keep a manifest row for corrupt/partial files.
            row["probe_error"] = f"{type(exc).__name__}: {exc}"
            failures.append(relative_path)
        rows.append(row)

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    labeled = [row for row in rows if row.get("parse_ok")]
    summary = {
        "root": str(args.root.resolve()),
        "video_count": len(rows),
        "parse_failures": [row["relative_path"] for row in rows if not row.get("parse_ok")],
        "probe_failures": failures,
        "total_size_bytes": sum(row.get("size_bytes", 0) for row in rows),
        "total_duration_s": round(sum(row.get("duration_s") or 0 for row in rows), 6),
        "participants": dict(sorted(Counter(row["participant"] for row in labeled).items())),
        "activities": dict(sorted(Counter(row["activity"] for row in labeled).items())),
        "lighting": dict(sorted(Counter(row["lighting"] for row in labeled).items())),
        "sequence_indices": dict(
            sorted(Counter(str(row["sequence_index"]) for row in labeled).items())
        ),
        "codecs": dict(sorted(Counter(row.get("video_codec") for row in rows).items(), key=str)),
        "resolutions": dict(
            sorted(Counter(f"{row.get('width')}x{row.get('height')}" for row in rows).items())
        ),
        "videos_with_audio": sum(bool(row.get("has_audio")) for row in rows),
        "videos_with_location_metadata": sum(
            bool(row.get("has_location_metadata")) for row in rows
        ),
        "videos_with_creation_time_metadata": sum(
            bool(row.get("has_creation_time_metadata")) for row in rows
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
