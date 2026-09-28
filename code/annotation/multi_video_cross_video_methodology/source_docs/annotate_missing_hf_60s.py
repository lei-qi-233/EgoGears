#!/usr/bin/env python3
"""Cut and caption the seven HuggingFace recordings that never got a 60-second pass.

The 60s layer covers 65 of the 72 HuggingFace recordings on disk. The seven left out are
not random: kit_scc_hochhaus has both its fast and normal runs missing, sinzheim_city both
its normal and slow -- while other speeds of the same scene were done. That pattern reads
as a batch that died partway and was never resumed, not as unusable footage. They do have
180s and whole-video descriptions, so the gap is specifically the fine granularity.

Reuses annotate_60s() from repair_runningbench_annotations.py so these segments come out
byte-compatible with the 1,342 that already exist -- same CAPTION_MODEL, same two-step
caption then structure pass, same five-field schema. Reimplementing the prompt would give
captions that read differently from every other row in the corpus.

Two things that module hardcodes for Google Drive have to be corrected here:
  - source_video_id is written as "google_drive/<participant>/<file>"; these are
    huggingface, and mislabelling provenance is what made the earlier source census wrong.
  - discover_sources() expects <root>/<participant>/<file>, but this footage nests one
    level deeper (Alec/sinzheim_city/sinzheim_city_slow.MOV), so the sources are passed
    in explicitly instead of discovered.

Usage:  annotate_missing_hf_60s.py [--dry-run] [--only STEM]
"""
import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

BASE = Path("/mnt/task_runtime/bolt/gdrive_relay")
HF_ROOT = Path("/mnt/data/cvhci_video_understanding/raw/huggingface")
WORK = Path("/mnt/data/data_anno/runningbench_hf_gapfill")
FFMPEG = Path("/mnt/data/data_anno/ffmpeg-7.0.2-amd64-static/ffmpeg")
FFPROBE = Path("/mnt/data/data_anno/ffmpeg-7.0.2-amd64-static/ffprobe")

# Explicit, because the layout nests deeper than discover_sources() expects.
MISSING = [
    "Markus/BLB-Euro_Fast_260518_200852.mp4",
    "Markus/Kiesweg_Fast_260518_202851.mp4",
    "Alec/kit_mensa_scc/kit_mensa_scc_slow.mov",
    "Alec/kit_scc_hochhaus/kit_scc_hochhaus_fast.mov",
    "Alec/kit_scc_hochhaus/kit_scc_hochhaus_normal.mov",
    "Alec/sinzheim_city/sinzheim_city_normal.mov",
    "Alec/sinzheim_city/sinzheim_city_slow.MOV",
]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="process just the recording whose stem matches")
    a = ap.parse_args()

    gen = load("gen", BASE / "repair_runningbench_annotations.py")

    sources = []
    for rel in MISSING:
        p = HF_ROOT / rel
        if not p.exists():
            print(f"  !! missing on disk: {rel}", flush=True)
            continue
        if a.only and a.only not in p.stem:
            continue
        sources.append(p)

    total_segments = 0
    for p in sources:
        secs = gen.duration(FFPROBE, p)
        n = int(-(-secs // 60))
        total_segments += n
        print(f"  {p.stem:44s} {secs:7.1f}s -> {n:2d} segments", flush=True)
    print(f"\n{len(sources)} recordings · {total_segments} segments", flush=True)
    if a.dry_run:
        return 0

    # Correct the provenance string the reused module writes. Patching the module's own
    # helper keeps annotate_60s() untouched, so its resume logic and schema stay intact.
    original_write = gen.write_json

    def write_json(path, payload):
        src = payload.get("original_video_path") or ""
        if "huggingface" in src:
            rel = os.path.relpath(src, HF_ROOT)
            payload["source_video_id"] = f"huggingface/{rel}"
            payload["provenance"] = "huggingface"
        return original_write(path, payload)

    gen.write_json = write_json

    api = gen.Floodgate(os.environ.get("FLOODGATE_PROJECT_TOKEN", ""))
    args = types.SimpleNamespace(work=WORK, ffmpeg=FFMPEG, ffprobe=FFPROBE)
    WORK.mkdir(parents=True, exist_ok=True)

    for i, source in enumerate(sources, 1):
        # The "participant" is only used to name output subdirectories; for this footage
        # the top-level HuggingFace folder (Alec / Markus) is the meaningful grouping.
        participant = os.path.relpath(source, HF_ROOT).split(os.sep)[0]
        print(f"\nsource {i}/{len(sources)}: {participant}/{source.name}", flush=True)
        try:
            gen.annotate_60s(args, api, source, participant)
        except Exception as exc:
            print(f"  !! {type(exc).__name__}: {exc}", flush=True)

    done = sorted(WORK.glob("annotations_60s/**/*.json"))
    print(f"\n{len(done)} annotations -> {WORK}/annotations_60s")
    ok = sum(1 for f in done if (json.loads(f.read_text()).get("dense_annotations")))
    print(f"  with dense_annotations: {ok}/{len(done)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
