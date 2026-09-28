#!/usr/bin/env python3
"""Package every question and caption in the corpus, both video sources, for human screening.

An earlier package covered only the Google Drive subset, because the annotators were
expected to have only that footage. The screening task is larger than that: of the 9,925
questions that have never been judged, 5,324 come from HuggingFace recordings and
Video Bench, so a Drive-only export hands the screeners under half of their own workload.

Deduplication matters here and is easy to get wrong. The corpus keeps six older copies of
the segment questions in separate directories -- conductor_release, gdrive_videos_annotations,
shaky_videos_annotations, and others -- with 10,566 rows between them and not one question
the current corpus lacks. Counting files instead of question texts inflates the total by
2.5x. Rows are keyed on a hash of the question text, first writer wins, and the authoritative
directory for each layer is listed first.

Provenance also cannot be read from a path prefix. Mikail and Markus are HuggingFace
subfolders whose older records still carry a /shaky_videos/ working-directory path, while
`shaky_videos_segmented/` is where clips from EVERY source were written -- so matching on
"shaky_videos" alone labels Google Drive recordings as HuggingFace. Only
original_video_path decides, and the raw-vs-segmented distinction is explicit.

Usage:  package_full_release.py [--out DIR]
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCENES = "/mnt/data/cvhci_video_understanding/metadata/route_manifest/scene_manifest.json"
DEFAULT_OUT = Path("/mnt/data/data_anno/runningbench_full_release")

# Authoritative directory per layer, in priority order. Anything not listed is an older
# copy: verified to contribute zero questions the current corpus does not already hold.
CAPTION_SOURCES = [
    ("segments_60s", ["/mnt/data/data_anno/runningbench_qa_expansion/annotations_60s/**/*.json",
                      "/mnt/data/data_anno/runningbench_hf_gapfill/annotations_60s/**/*.json"]),
    ("windows_180s", ["/mnt/data/data_anno/runningbench_gap_repair/annotations_180s/**/*.json"]),
    ("windows_180s_aggregated",
                     ["/mnt/data/data_anno/runningbench_aggregate_fill/annotations_180s/**/*.json"]),
    ("whole_video",  ["/mnt/data/data_anno/runningbench_gap_repair/annotations_whole/**/*.json"]),
    ("whole_video_aggregated",
                     ["/mnt/data/data_anno/runningbench_aggregate_fill/annotations_whole/**/*.json"]),
]


def qhash(text):
    return hashlib.sha256((text or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def gdrive_stems():
    return {os.path.basename(r["file"]).rsplit(".", 1)[0]
            for r in json.load(open(SCENES))["runs"] if "google_drive" in (r.get("file") or "")}


def provenance(path):
    """Google Drive or HuggingFace, from the ORIGINAL path only.

    `/shaky_videos/Mikail/` is HuggingFace footage under its old working-directory path;
    `/shaky_videos_segmented/` is the clip output directory shared by all sources and says
    nothing about origin. Testing for "shaky_videos" without that distinction is what made
    an earlier census report four sources where there are two.
    """
    if "google_drive" in path or "GDrive_Data" in path:
        return "google_drive"
    if "huggingface" in path.lower():
        return "huggingface"
    if re.search(r"/shaky_videos/(Mikail|Markus|Alec)/", path):
        return "huggingface"
    return "unknown"


def load_items(path):
    try:
        doc = json.load(open(path))
    except Exception:
        return None, []
    if not isinstance(doc, dict):
        return None, []
    items = []
    for field in ("qa_pairs", "questions"):
        v = doc.get(field)
        if isinstance(v, list):
            items += [x for x in v if isinstance(x, dict)]
    return doc, items


def is_objective(item):
    opts = item.get("options") or item.get("choices")
    ans = item.get("answer")
    has = (isinstance(opts, dict) and opts) or (isinstance(opts, list) and opts)
    return bool(has) and ans not in (None, "", [])


def gate_status(item):
    g = item.get("gates") or {}
    if not g:
        return "ungated", None, None
    blind = bool((g.get("blind_guess") or {}).get("matches_gold"))
    rc = g.get("visual_recheck") or {}
    verdict = rc.get("verdict") if isinstance(rc, dict) else rc
    if blind:
        return "rejected_blind_guessable", verdict, (g.get("blind_guess") or {}).get("hits_of_3")
    if verdict == "supported":
        return "usable", verdict, (g.get("blind_guess") or {}).get("hits_of_3")
    return f"rejected_{verdict}", verdict, (g.get("blind_guess") or {}).get("hits_of_3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    out = Path(a.out)
    gd = gdrive_stems()
    seen = set()
    stats = collections.OrderedDict()

    def write(rel, rows):
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[rel] = {"rows": len(rows), "bytes": p.stat().st_size}
        return len(rows)

    # ---------------- captions ----------------
    for label, pats in CAPTION_SOURCES:
        rows = []
        for pat in pats:
            for f in sorted(glob.glob(pat, recursive=True)):
                doc, _ = load_items(f)
                if not doc:
                    continue
                cap = doc.get("dense_annotations") or doc.get("annotation_raw")
                if not cap:
                    continue
                src = doc.get("original_video_path") or ""
                video = os.path.basename(src).rsplit(".", 1)[0]
                idx = doc.get("segment_index", doc.get("chunk_index", 0))
                key = (label, video, idx)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "id": f"{video}_{label}_{idx}", "video": video,
                    "provenance": provenance(src), "index": idx,
                    "span_sec": [doc.get("start_time_sec"), doc.get("end_time_sec")],
                    "model": doc.get("model_name"),
                    "aggregated_without_video": bool(doc.get("aggregated_without_video")),
                    "derived_from": [os.path.basename(x) for x in (doc.get("derived_from") or [])],
                    "caption": cap,
                })
        write(f"captions/{label}.jsonl", rows)

    # whole-video structural annotations (the full-video line's own pass)
    rows = []
    for f in sorted(glob.glob("/mnt/data/data_anno/runningbench_fullvideo/**/annotations/*.json",
                              recursive=True)):
        rec = json.load(open(f))
        stem = os.path.basename(f).rsplit(".", 1)[0]
        rows.append({"id": stem, "video": stem,
                     "provenance": "google_drive" if stem in gd else "huggingface",
                     "scene_id": rec.get("scene_id"), "route_id": rec.get("route_id"),
                     "duration_sec": rec.get("duration_sec"), "model": rec.get("global_model"),
                     "run_meta": rec.get("run_meta"),
                     "route_phases": rec.get("route_phases"),
                     "environment_stages": rec.get("environment_stages"),
                     "dynamic_events": rec.get("dynamic_events"),
                     "revisited_landmarks": rec.get("revisited_landmarks"),
                     "start_end_relation": rec.get("start_end_relation")})
    write("captions/fullvideo_structural.jsonl", rows)

    # ---------------- questions ----------------
    qseen = set()
    tally = collections.Counter()

    def emit(item, doc, unit, layer, video_ids=None, forced_provenance=None):
        h = qhash(item.get("question"))
        if h in qseen:
            return None
        qseen.add(h)
        opts = item.get("options") or item.get("choices")
        ans = item.get("answer")
        status, verdict, hits = gate_status(item)
        src = (doc or {}).get("original_video_path") or ""
        if video_ids:
            prov = "google_drive" if all(v in gd for v in video_ids) else "huggingface"
            video = video_ids if len(video_ids) > 1 else video_ids[0]
        else:
            prov = forced_provenance or provenance(src)
            video = os.path.basename(src).rsplit(".", 1)[0] if src else None
        n_ans = len(ans) if isinstance(ans, list) else 1
        row = {
            "id": h, "layer": layer, "unit": unit, "provenance": prov, "video": video,
            "span_sec": [(doc or {}).get("start_time_sec"), (doc or {}).get("end_time_sec")],
            "question": item.get("question"),
            "question_type": item.get("question_type") or item.get("type"),
            "options": opts, "answer": ans,
            "arity": "multi" if n_ans > 1 else "single",
            "n_options": len(opts) if isinstance(opts, (dict, list)) else None,
            "option_order": item.get("option_order") or (doc or {}).get("option_order") or "unknown",
            "evidence_spans": item.get("evidence_spans"),
            "why_hard": item.get("why_hard"),
            "screening_status": status,
            "gates": ({"blind_guess_hits_of_3": hits, "visual_recheck": verdict,
                       "recheck_notes": ((item.get("gates") or {}).get("visual_recheck") or {}).get("notes")
                       if isinstance((item.get("gates") or {}).get("visual_recheck"), dict) else None}
                      if item.get("gates") else None),
        }
        tally[(layer, status)] += 1
        tally[(prov, "total")] += 1
        return row

    seg = []
    for pat in ("/mnt/data/data_anno/runningbench_qa_expansion/annotations_60s/**/*.json",
                "/mnt/data/data_anno/runningbench_hf_gapfill/annotations_60s/**/*.json"):
        for f in sorted(glob.glob(pat, recursive=True)):
            doc, items = load_items(f)
            for it in items:
                if not is_objective(it):
                    continue
                r = emit(it, doc, "60s_segment", "segment_60s")
                if r:
                    seg.append(r)
    write("questions/segments_60s.jsonl", seg)

    fv, cv = [], []
    for f in sorted(glob.glob("/mnt/data/data_anno/runningbench_fullvideo/route_*/questions/*.json")):
        doc, items = load_items(f)
        for it in items:
            if not is_objective(it):
                continue
            raw = it.get("video_id") or it.get("video_ids")
            vids = ([raw] if isinstance(raw, str)
                    else list(raw.values()) if isinstance(raw, dict) else list(raw or []))
            if not vids:
                continue
            unit = "cross_video" if len(vids) > 1 else "whole_video"
            r = emit(it, doc, unit, "fullvideo", vids)
            if r:
                r["video_ids"] = raw if isinstance(raw, dict) else None
                (cv if unit == "cross_video" else fv).append(r)
    write("questions/whole_video.jsonl", fv)
    write("questions/cross_video.jsonl", cv)

    exc = []
    for f in sorted(glob.glob("/mnt/data/data_anno/runningbench_excerpts/fullvideo/*/questions/*.json")):
        doc, items = load_items(f)
        for it in items:
            if not is_objective(it):
                continue
            raw = it.get("video_id") or it.get("video_ids")
            vids = ([raw] if isinstance(raw, str)
                    else list(raw.values()) if isinstance(raw, dict) else list(raw or []))
            r = emit(it, doc, "excerpt", "excerpt", vids or None)
            if r:
                exc.append(r)
    write("questions/excerpts.jsonl", exc)

    vb = []
    for pat in ("/mnt/data/data_anno/videobench_rebalanced/videobench_main.json",
                "/mnt/data/data_anno/videobench_rebalanced/per_video/**/*.json"):
        for f in sorted(glob.glob(pat, recursive=True)):
            doc, items = load_items(f)
            for it in items:
                if not is_objective(it):
                    continue
                # Video Bench rows carry no original_video_path, so provenance() returns
                # "unknown"; the footage is the HuggingFace set. Fix it before the tally
                # counts it, not after -- the manifest reads the counter, not these rows.
                r = emit(it, doc, "25s_segment", "videobench", forced_provenance="huggingface")
                if r:
                    r["family"] = it.get("family")
                    r["difficulty"] = it.get("difficulty")
                    r["answer_format"] = it.get("answer_format")
                    vb.append(r)
    write("questions/videobench.jsonl", vb)

    # repaired distractors, kept apart: same questions, new options, never re-tested
    rep = []
    src = "/mnt/data/data_anno/runningbench_segment_repair/repaired_multi.json"
    if os.path.exists(src):
        for x in json.load(open(src))["repaired"]:
            it = x["item"]
            rep.append({"id": qhash(it.get("question")), "question": it.get("question"),
                        "options": it.get("options"), "answer": it.get("answer"),
                        "distractors_rebuilt": it.get("distractors_rebuilt", 0),
                        "fully_reverted": it.get("distractors_rebuilt", 0) == 0,
                        "screening_status": "ungated",
                        "note": "distractors rebuilt from the text annotation; no video, no gates"})
    write("questions/segments_60s_repaired_distractors.jsonl", rep)

    total_q = len(seg) + len(fv) + len(cv) + len(exc) + len(vb)
    by_status = collections.Counter()
    for (layer, status), n in tally.items():
        if status in ("total",):
            continue
        by_status[status] += n

    manifest = {
        "release": "runningbench-full-v1",
        "sources": ["google_drive", "huggingface"],
        "questions_total_deduplicated": total_q,
        "by_status": dict(by_status),
        "by_provenance": {k: v for (k, s), v in tally.items() if s == "total"},
        "files": dict(stats),
        "note": "Question rows are deduplicated on question text. Older duplicate copies of "
                "the segment corpus (conductor_release, gdrive_videos_annotations, "
                "shaky_videos_annotations*) contribute nothing and are excluded.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    print(f"{'file':52s} {'rows':>7s} {'size':>9s}")
    for k, v in stats.items():
        print(f"  {k:50s} {v['rows']:7d} {v['bytes']/2**20:8.2f}M")
    print(f"\nquestions {total_q} · by status {dict(by_status)}")
    print(f"by provenance {manifest['by_provenance']}")
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
