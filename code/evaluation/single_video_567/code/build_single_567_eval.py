#!/usr/bin/env python3
"""Build the auditable 567-question single-video evaluation corpus."""

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("~/datasets/RunningBench").expanduser())
    parser.add_argument("--single-clips", type=Path,
                        default=Path("~/datasets/seg600_verification/videoclips").expanduser())
    args = parser.parse_args()
    root = args.root.expanduser()
    clips = args.single_clips.expanduser()
    source = root / "human_answer_repair_567" / "runningbench_567_human_usable.jsonl"
    corrected = root / "human_answer_repair_567" / "runningbench_qa_human_corrected.jsonl"
    ledger = root / "human_answer_repair_567" / "runningbench_567_inclusion_ledger.csv"
    reference = root / "segments_60s_gemini_verification" / "final" / "segments_700_verified.jsonl"
    destination = root / "human_answer_repair_567" / "single_567_human_eval.jsonl"
    manifest_path = root / "human_answer_repair_567" / "single_567_human_eval_manifest.json"

    rows = list(read_jsonl(source))
    original = {row["id"]: row for row in read_jsonl(reference)}
    corrected_rows = list(read_jsonl(corrected))
    corrected_by_id = {row["question_id"]: row for row in corrected_rows}
    with ledger.open(encoding="utf-8-sig", newline="") as stream:
        ledger_rows = list(csv.DictReader(stream))
    if len(rows) != 567 or len(original) != 698 or len(corrected_rows) != 698 or len(ledger_rows) != 567:
        raise ValueError("Unexpected source, reference, corrected, or ledger size")
    ids = [row["question_id"] for row in rows]
    if len(set(ids)) != 567:
        raise ValueError("The 567 source contains duplicate question IDs")
    if set(ids) != {row["question_id"] for row in ledger_rows}:
        raise ValueError("567 source question IDs do not match the inclusion ledger")

    missing_reference = [qid for qid in ids if qid not in original]
    missing_media = [qid for qid in ids if not (clips / f"{qid}.mp4").is_file()]
    if missing_reference or missing_media:
        raise ValueError(f"Missing reference IDs={missing_reference[:10]} media={missing_media[:10]}")

    output = []
    changes = collections.Counter()
    sources = collections.Counter()
    n_select = collections.Counter()
    for row in rows:
        qid = row["question_id"]
        prior = original[qid]
        amended = corrected_by_id[qid]
        if any(row[key] != amended[key] for key in ("question", "options", "answer")):
            raise ValueError(f"567 row differs from corrected 698 gold for {qid}")
        if row["video_id"] != prior["video"]:
            raise ValueError(f"Video ID mismatch for {qid}: {row['video_id']} != {prior['video']}")
        span = [float(row["start_second"]), float(row["end_second"])]
        if span != [float(x) for x in prior["span_sec"]]:
            raise ValueError(f"Video time span mismatch for {qid}: {span} != {prior['span_sec']}")
        options = row["options"]
        answer = row["answer"]
        if not isinstance(options, dict) or not options:
            raise ValueError(f"Invalid options for {qid}")
        if not isinstance(answer, list) or not answer or len(answer) != len(set(answer)):
            raise ValueError(f"Invalid answer for {qid}")
        if any(letter not in options for letter in answer):
            raise ValueError(f"Gold answer absent from options for {qid}")
        if not row["question"].strip():
            raise ValueError(f"Empty question for {qid}")
        changes["question_changed"] += row["question"] != prior["question"]
        changes["options_changed"] += options != prior["options"]
        changes["answer_changed"] += set(answer) != set(prior["answer"])
        sources[row["usability_source"]] += 1
        n_select[len(answer)] += 1
        output.append({
            "id": qid,
            "unit": "60s_segment",
            "question": row["question"],
            "options": options,
            "answer": answer,
            "question_type": prior.get("question_type"),
            "category": row.get("category"),
            "video": row["video_id"],
            "span_sec": span,
            "segment": row.get("segment"),
            # Relative to the release root, so the Hala copy and the HF package are byte-identical.
            "media_file": f"videos/{qid}.mp4",
            "usability_source": row["usability_source"],
            "repair_status": row.get("repair_status", ""),
        })

    temp = destination.with_suffix(".jsonl.part")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, destination)
    manifest = {
        "dataset": "single_567_human",
        "questions": len(output),
        "media_available": len(output),
        "source": str(source),
        "source_sha256": sha256(source),
        "corrected_698": str(corrected),
        "corrected_698_sha256": sha256(corrected),
        "inclusion_ledger": str(ledger),
        "inclusion_ledger_sha256": sha256(ledger),
        "reference": str(reference),
        "reference_sha256": sha256(reference),
        "eval_jsonl": str(destination),
        "eval_sha256": sha256(destination),
        "usability_source_counts": dict(sources),
        "answer_count_distribution": dict(sorted(n_select.items())),
        "changes_from_original_698": dict(changes),
        "note": "Gold/question/options come from the human usable subset, not the original 698."
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
