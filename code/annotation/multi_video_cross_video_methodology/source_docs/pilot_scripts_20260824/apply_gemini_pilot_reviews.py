#!/usr/bin/env python3
"""Merge manual pilot reviews while retaining raw proposals for auditability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ACTIVITY_MAP = {"FastWalk": "fast_walk", "SlowWalk": "slow_walk", "Running": "running"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    overrides = json.loads(args.reviews.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.annotations.read_text(encoding="utf-8").splitlines() if line]
    activity_matches = 0
    lighting_matches = 0
    rejected_events = 0
    rejected_qa = 0
    reviewed = []
    for record in records:
        review = overrides[record["annotation_id"]]
        rejected_event_indices = set(review["rejected_timeline_indices"])
        rejected_qa_indices = set(review["rejected_qa_indices"])
        timeline = record["proposal"]["timeline"]
        qa_pairs = record["proposal"]["qa_pairs"]
        record["review"] = review
        record["accepted_view"] = {
            "timeline": [value for index, value in enumerate(timeline) if index not in rejected_event_indices],
            "qa_pairs": [value for index, value in enumerate(qa_pairs) if index not in rejected_qa_indices],
        }
        source_activity = ACTIVITY_MAP[record["source"]["activity"]]
        source_lighting = record["source"]["lighting"].lower()
        activity_match = record["proposal"]["movement"]["label"] == source_activity
        lighting_match = record["proposal"]["lighting"] == source_lighting
        record["review"]["source_label_comparison"] = {
            "activity_exact_match": activity_match,
            "lighting_exact_match": lighting_match,
        }
        activity_matches += int(activity_match)
        lighting_matches += int(lighting_match)
        rejected_events += len(rejected_event_indices)
        rejected_qa += len(rejected_qa_indices)
        reviewed.append(record)

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in reviewed:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)

    audit = {
        "schema": "cvhci_gemini_video_annotation_pilot_review_audit_v1",
        "records": len(reviewed),
        "review_scope": "manual_contact_sheet_spot_check_not_full_frame_exhaustive_review",
        "activity_exact_matches": activity_matches,
        "activity_exact_match_rate": activity_matches / len(reviewed),
        "lighting_exact_matches": lighting_matches,
        "lighting_exact_match_rate": lighting_matches / len(reviewed),
        "raw_timeline_events": sum(len(record["proposal"]["timeline"]) for record in reviewed),
        "rejected_timeline_events": rejected_events,
        "accepted_timeline_events": sum(len(record["accepted_view"]["timeline"]) for record in reviewed),
        "raw_qa_pairs": sum(len(record["proposal"]["qa_pairs"]) for record in reviewed),
        "rejected_qa_pairs": rejected_qa,
        "accepted_qa_pairs": sum(len(record["accepted_view"]["qa_pairs"]) for record in reviewed),
        "gold_ready": False,
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
