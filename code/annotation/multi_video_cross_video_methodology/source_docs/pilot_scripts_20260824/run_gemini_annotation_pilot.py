#!/usr/bin/env python3
"""Create clips, run Gemini proposals, and assemble a reviewable JSONL pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-python", type=Path, required=True)
    parser.add_argument("--api-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", default="gemini-3.6-flash")
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    clips = args.output_root / "clips"
    results = args.output_root / "results"
    review_sheets = args.output_root / "review_sheets"
    for directory in (clips, results, review_sheets):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    for index, item in enumerate(plan, start=1):
        annotation_id = f"pilot_{index:03d}_{item['source_id']}"
        source = args.source_root / item["source_relative_path"]
        clip = clips / f"{annotation_id}.mp4"
        result_path = results / f"{annotation_id}.json"
        review_sheet = review_sheets / f"{annotation_id}.jpg"
        if not clip.exists():
            run(
                [
                    str(args.clip_python),
                    str(scripts / "make_video_clip.py"),
                    str(source),
                    str(clip),
                    "--start-s",
                    str(item["clip_start_s"]),
                    "--duration-s",
                    str(item["clip_duration_s"]),
                    "--fps",
                    "8",
                    "--width",
                    "640",
                    "--height",
                    "360",
                ]
            )
        if not result_path.exists():
            run(
                [
                    str(args.api_python),
                    str(scripts / "gemini_video_pilot.py"),
                    str(clip),
                    "--duration-s",
                    str(item["clip_duration_s"]),
                    "--model",
                    args.model,
                    "--output",
                    str(result_path),
                ]
            )
        if not review_sheet.exists():
            run(
                [
                    str(args.clip_python),
                    str(scripts / "make_contact_sheet.py"),
                    str(clip),
                    "--positions",
                    "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
                    "--output",
                    str(review_sheet),
                ]
            )

        gemini = json.loads(result_path.read_text(encoding="utf-8"))
        records.append(
            {
                "schema": "cvhci_gemini_video_annotation_proposal_v1",
                "annotation_id": annotation_id,
                "annotation_status": "proposal_requires_review",
                "model_received_source_labels": False,
                "source_labels_authoritative": True,
                "source": item,
                "clip": {
                    "relative_path": clip.relative_to(args.output_root).as_posix(),
                    **gemini["video"],
                },
                "model": gemini["model"],
                "latency_s": gemini["latency_s"],
                "usage_metadata": gemini.get("usage_metadata"),
                "proposal": gemini["result"],
                "review_sheet": review_sheet.relative_to(args.output_root).as_posix(),
                "review": {
                    "status": "pending",
                    "unsupported_claims": [],
                    "timestamp_issues": [],
                    "notes": [],
                },
            }
        )

    output = args.output_root / "annotations.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)
    summary = {
        "schema": "cvhci_gemini_video_annotation_pilot_summary_v1",
        "model": args.model,
        "records": len(records),
        "status": "proposal_requires_review",
        "total_latency_s": round(sum(record["latency_s"] for record in records), 3),
        "participants": sorted({record["source"]["participant"] for record in records}),
        "activities": sorted({record["source"]["activity"] for record in records}),
        "lighting": sorted({record["source"]["lighting"] for record in records}),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
