#!/usr/bin/env python3
"""Package the Google-Drive-sourced captions and QA pairs as one release directory.

The corpus draws on four separate video sources -- a controlled Google Drive collection
plus footage from HuggingFace and two contributors' folders -- and only the Drive subset
carries the structural metadata (scene_id, route_id, turnaround_sec) that the route
questions are built and checked against. This exports that subset alone, so a reader is
never left guessing which rows have route metadata behind them and which do not.

Provenance is taken from scene_manifest.json, not from filename shape: the run list there
is the authority on which recording came from where, and inferring it from a "P01_" prefix
would silently include or drop files the manifest disagrees with.

Everything is written as JSONL, one record per line, so a 4,764-row question file can be
streamed instead of loaded whole.

Usage:  package_gdrive_release.py [--out DIR] [--copy-clips]
"""
import argparse
import collections
import json
import glob
import os
import shutil
import sys
from pathlib import Path

SCENES = "/mnt/data/cvhci_video_understanding/metadata/route_manifest/scene_manifest.json"
SEG60 = "/mnt/data/data_anno/runningbench_qa_expansion/annotations_60s"
GAP = "/mnt/data/data_anno/runningbench_gap_repair"
FULL = "/mnt/data/data_anno/runningbench_fullvideo"
DEFAULT_OUT = Path("/mnt/data/data_anno/runningbench_gdrive_release")


def gdrive_runs():
    """run_id -> run record, for the Google Drive subset only."""
    out = {}
    for r in json.load(open(SCENES))["runs"]:
        if "google_drive" in (r.get("file") or ""):
            out[os.path.basename(r["file"]).rsplit(".", 1)[0]] = r
    return out


def stem_of(rec):
    p = rec.get("original_video_path") or rec.get("video_path") or ""
    return os.path.basename(p).rsplit(".", 1)[0] if p else None


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), path.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--copy-clips", action="store_true",
                    help="also copy the 480p re-check clips (adds several GiB)")
    a = ap.parse_args()
    out = Path(a.out)
    runs = gdrive_runs()
    print(f"Google Drive subset: {len(runs)} recordings", flush=True)

    stats = collections.OrderedDict()

    # ---- 60 s dense captions + their multiple-choice questions ----
    cap60, qa60 = [], []
    for f in sorted(glob.glob(f"{SEG60}/**/*.json", recursive=True)):
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        stem = stem_of(rec)
        if stem not in runs:
            continue
        seg = rec.get("segment_index")
        cap60.append({
            "id": f"{stem}_seg{seg}",
            "video": stem, "segment_index": seg,
            "span_sec": [rec.get("start_time_sec"), rec.get("end_time_sec")],
            "model": rec.get("model_name"),
            "caption": rec.get("dense_annotations") or rec.get("annotation_raw"),
        })
        for i, item in enumerate(rec.get("qa_pairs") or []):
            opts, ans = item.get("options"), item.get("answer")
            if not (isinstance(opts, dict) and opts and isinstance(ans, list) and ans):
                continue
            qa60.append({
                "id": f"{stem}_seg{seg}_q{i}",
                "video": stem, "segment_index": seg,
                "span_sec": [rec.get("start_time_sec"), rec.get("end_time_sec")],
                "question": item.get("question"),
                "options": opts, "answer": ans,
                "arity": "single" if len(ans) == 1 else "multi",
                "type": item.get("type"),
                "option_order": rec.get("option_order", "unknown"),
            })
    stats["captions/segments_60s.jsonl"] = write_jsonl(out / "captions/segments_60s.jsonl", cap60)
    stats["qa/segments_60s.jsonl"] = write_jsonl(out / "qa/segments_60s.jsonl", qa60)

    # ---- 180 s windows and whole-video open-ended descriptions ----
    for label, sub, gran in (("windows_180s", "annotations_180s", "180s"),
                             ("whole_video", "annotations_whole", "whole")):
        caps, qas = [], []
        for f in sorted(glob.glob(f"{GAP}/{sub}/**/*.json", recursive=True)):
            try:
                rec = json.load(open(f))
            except Exception:
                continue
            stem = stem_of(rec)
            if stem not in runs:
                continue
            idx = rec.get("chunk_index", 0)
            caps.append({
                "id": f"{stem}_{gran}{idx}", "video": stem, "granularity": gran,
                "chunk_index": idx,
                "span_sec": [rec.get("start_time_sec"), rec.get("end_time_sec")],
                "model": rec.get("model_name"),
                "caption": rec.get("dense_annotations") or rec.get("annotation_raw"),
                "derived_from_60s": [os.path.basename(x) for x in (rec.get("derived_from") or [])],
            })
            for i, item in enumerate(rec.get("qa_pairs") or []):
                qas.append({
                    "id": f"{stem}_{gran}{idx}_q{i}", "video": stem, "granularity": gran,
                    "span_sec": [rec.get("start_time_sec"), rec.get("end_time_sec")],
                    "question": item.get("question"),
                    "answer": item.get("answer"),      # free text, no options
                    "type": item.get("type"), "format": "open_ended",
                })
        stats[f"captions/{label}.jsonl"] = write_jsonl(out / f"captions/{label}.jsonl", caps)
        stats[f"qa/{label}_open.jsonl"] = write_jsonl(out / f"qa/{label}_open.jsonl", qas)

    # ---- whole-video structural annotations and their gated questions ----
    fv_caps = []
    for f in sorted(glob.glob(f"{FULL}/**/annotations/*.json", recursive=True)):
        rec = json.load(open(f))
        stem = os.path.basename(f).rsplit(".", 1)[0]
        if stem not in runs:
            continue
        fv_caps.append({
            "id": stem, "video": stem,
            "scene_id": rec.get("scene_id"), "route_id": rec.get("route_id"),
            "duration_sec": rec.get("duration_sec"),
            "model": rec.get("global_model"),
            "run_meta": rec.get("run_meta"),
            "route_phases": rec.get("route_phases"),
            "environment_stages": rec.get("environment_stages"),
            "dynamic_events": rec.get("dynamic_events"),
            "revisited_landmarks": rec.get("revisited_landmarks"),
            "start_end_relation": rec.get("start_end_relation"),
        })
    stats["captions/fullvideo_structural.jsonl"] = write_jsonl(
        out / "captions/fullvideo_structural.jsonl", fv_caps)

    fv_qa = []
    for f in sorted(glob.glob(f"{FULL}/**/questions/*.json", recursive=True)):
        doc = json.load(open(f))
        for q in doc.get("questions") or []:
            # Whole-video questions carry `video_id`; cross-video ones carry `video_ids`
            # (with `required_video_ids` alongside) and leave `video_id` null. Reading only
            # the singular field silently drops every cross-video question -- all 347 of
            # them -- which is the one part of this corpus nothing else replaces.
            # Three shapes in one corpus. A whole-video question carries `video_id` as a
            # string. A cross-video question leaves that null and carries `video_ids` as a
            # LABEL->recording map ({"M": "P03_FastWalk_...", ...}) with `required_video_ids`
            # listing the labels, not the recordings -- so reading either as a list of
            # recordings yields label letters and matches nothing. Both earlier attempts
            # dropped all 347 cross-video questions, the one part of this corpus that
            # nothing else replaces.
            raw = q.get("video_id") or q.get("video_ids")
            if isinstance(raw, str):
                vids, labels = [raw], None
            elif isinstance(raw, dict):
                vids, labels = list(raw.values()), raw
            else:
                vids, labels = list(raw or []), None
            # A cross-video question belongs to the release only if every recording it
            # compares is in the subset; a partial one cannot be answered from what ships.
            if not vids or any(v not in runs for v in vids):
                continue
            g = q.get("gates") or {}
            rc = g.get("visual_recheck")
            verdict = rc.get("verdict") if isinstance(rc, dict) else rc
            blind = bool((g.get("blind_guess") or {}).get("matches_gold"))
            fv_qa.append({
                "id": f"{'+'.join(vids)}_{abs(hash(q.get('question'))) % 10**8}",
                "video_id": q.get("video_id"),
                "video_ids": labels,
                "scene_id": q.get("scene_id"), "route_id": q.get("route_id"),
                "unit": "cross_video" if len(vids) > 1 else "whole_video",
                "question": q.get("question"), "question_type": q.get("question_type"),
                "options": q.get("options"), "answer": q.get("answer"),
                "option_order": q.get("option_order"),
                "evidence_spans": q.get("evidence_spans"),
                "why_hard": q.get("why_hard"),
                "gates": {"blind_guess_hits_of_3": (g.get("blind_guess") or {}).get("hits_of_3"),
                          "blind_guessable": blind,
                          "visual_recheck": verdict,
                          "recheck_notes": rc.get("notes") if isinstance(rc, dict) else None},
                "usable": verdict == "supported" and not blind,
            })
    stats["qa/fullvideo_gated.jsonl"] = write_jsonl(out / "qa/fullvideo_gated.jsonl", fv_qa)

    if a.copy_clips:
        dst = out / "clips_480p_180s"
        n = 0
        for src in glob.glob("/mnt/data/data_anno/runningbench_windows/clips_480p/**/*.mp4", recursive=True):
            stem = os.path.basename(os.path.dirname(src))
            if stem not in runs:
                continue
            (dst / stem).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / stem / os.path.basename(src))
            n += 1
        print(f"copied {n} clips", flush=True)

    manifest = {
        "release": "runningbench-gdrive-v1",
        "source": "Google Drive controlled collection (scene_manifest.json subset)",
        "recordings": len(runs),
        "total_hours": round(sum(r.get("duration_sec") or 0 for r in runs.values()) / 3600, 2),
        "participants": sorted({r.get("participant") for r in runs.values() if r.get("participant")}),
        "files": {k: {"rows": v[0], "bytes": v[1]} for k, v in stats.items()},
        "usable_fullvideo_questions": sum(1 for q in fv_qa if q["usable"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    print(f"\n{'file':44s} {'rows':>7s} {'size':>10s}")
    for k, (rows, size) in stats.items():
        print(f"  {k:42s} {rows:7d} {size/2**20:9.2f}M")
    print(f"\n{len(runs)} recordings · {manifest['total_hours']} h · "
          f"participants {manifest['participants']}")
    print(f"usable gated full-video questions: {manifest['usable_fullvideo_questions']}")
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
