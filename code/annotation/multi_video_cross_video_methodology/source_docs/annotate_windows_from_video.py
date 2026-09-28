#!/usr/bin/env python3
"""Annotate the 180-second windows by WATCHING them, then build questions from that.

Why this replaces the previous attempt. The first pass wrote questions from the existing
180s annotations, which are aggregations of three 60s captions -- `derived_from` names
them -- and the pilot came back at 68.4% blind-guessable (CI [57.3, 77.8]), worse than the
segment corpus it was meant to improve on. The re-check gate was fine: only 3 of 76 were
contradicted. The questions were simply easy to guess, and by question type the failure was
total where it should have been strongest -- window-turn-sequence scored 0/19.

Aggregation is what broke it. Merging three minute-long captions flattens exactly what a
three-minute question needs: a pedestrian in a blue bucket hat becomes "a pedestrian", a
specific turn at a specific marker becomes "the path curves". Generic ground truth can only
produce generic options, and then "which option sounds most coherent" is a winning strategy
without watching anything.

So the annotation is made from the footage at the scale the questions are asked at. Two
things carry over from the full-video pipeline, which reaches 7.3% blind on 8-option multi:

  1. Its structural schema -- route_phases, turns, revisited_landmarks, environment_stages,
     dynamic_events -- rather than the 60s pass's frame-level five fields. Order and
     progression are what three minutes supports asking about.
  2. A clock burned into the top-left corner, one frame per second, showing true recording
     time. The model reads timestamps instead of estimating them; the full-video pass found
     estimates drifting +240 s over a 448 s video, and a question about what happened first
     is worthless if the annotation's own ordering is guesswork.

Usage:
  annotate_windows_from_video.py --clips-only            # re-cut with the clock, no API
  annotate_windows_from_video.py --pilot 8               # annotate + questions + gates
  annotate_windows_from_video.py --all --jobs 2
"""
import argparse
import base64
import collections
import concurrent.futures
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

BASE = Path("/mnt/task_runtime/bolt/gdrive_relay")
SRC = "/mnt/data/data_anno/runningbench_gap_repair/annotations_180s"
OUT = Path("/mnt/data/data_anno/runningbench_windows_v2")
CLIPS = OUT / "clips_clock_480p"
FFMPEG = "/mnt/data/data_anno/ffmpeg-7.0.2-amd64-static/ffmpeg"
INLINE_CAP = 11 * 1024 * 1024
QUOTA_PER_SECOND = 0.45
SEED = 20260831

TYPES = ("window-event-order", "window-turn-sequence",
         "window-environment-shift", "window-object-timing")

LEAK = ("trial", "traj", ".mov", ".mp4", "filename", "annotation",
        "described", "the description", "the evidence", "per the", "according to the")

OUT_OF_SCENE = ("beach", "highway", "motorway", "forest", "woods", "indoor", "shopping mall",
                "airport terminal", "subway", "train station", "stadium", "boardwalk",
                "desert", "mountain trail", "swimming", "driving", "dashboard", "cockpit")

ANNOTATE_PROMPT = """You are watching a {length}-second stretch of a first-person walking/running recording,
anonymous id {vid}. The stretch covers {p0}-{p1} seconds of the full recording. Do not infer
anything from a filename -- you have none.

A clock in the TOP-LEFT corner shows the true recording time as MM:SS. READ every time you
report from that clock and convert to seconds; never estimate time from pacing.

Your job is the STRUCTURE of this stretch, in order, and the DISTINCTIVE detail that
identifies each place. Both matter: order alone gives questions no grounding, and detail
alone gives no order to ask about.

Return one JSON object:
{{
 "route_phases": [ {{"start_sec": {p0}, "end_sec": 0, "description": "...",
                     "landmarks": ["specific, identifying, not 'a building'"],
                     "surface": "asphalt|gravel|paved|dirt|mixed", "setting": "..."}} ],
 "turns": [ {{"time_sec": 0, "direction": "left|right|u-turn|straight-on",
              "sharpness": "gentle|moderate|sharp", "evidence": "what marks it"}} ],
 "revisited_landmarks": [ {{"landmark": "...", "first_sec": 0, "second_sec": 0,
                            "direction_change": "same|opposite|unclear"}} ],
 "environment_stages": ["how the surroundings progress, in order"],
 "dynamic_events": [ {{"time_sec": 0, "description": "...",
                       "kind": "person|vehicle|animal|obstacle|other"}} ],
 "distinctive_objects": [ {{"object": "...", "attributes": ["colour", "material", "..."],
                            "side": "left|right|ahead|overhead", "seen_sec": 0}} ],
 "uncertainties": ["..."]
}}

Phases must tile {p0}-{p1} seconds in order: each end_sec equals the next start_sec, the
first start_sec is {p0}, the last end_sec is {p1}.

Name landmarks and objects specifically enough that someone could recognise this exact
place -- "a red and white striped warning pole", not "a pole". Use only what is visible."""

Q_PROMPT = """Write exactly {n} benchmark questions about ONE {length}-second stretch of a first-person
walking/running video, anonymous id {vid}, covering {p0}-{p1} seconds of the recording.

ANNOTATION (the only ground truth; it was written while watching this exact stretch):
{annotation}

Each question must need the WHOLE stretch. A question answerable from one frame, or from any
single minute of it, does not belong here: ask about order, about change across the stretch,
about what happened before or after what.

Use these types, spread evenly: {types}
  window-event-order        the sequence in which events occur across the stretch
  window-turn-sequence      the trajectory: which turns, in which order, how sharp
  window-environment-shift  how the surroundings change from the start to the end
  window-object-timing      which objects appear in which part of the stretch

FORMAT -- exactly 8 options, of which exactly 3 are correct. Return the three correct ones
first; they get shuffled afterwards.

THE FIVE WRONG OPTIONS ARE THE HARD PART. An earlier version of this task returned 68% of
its questions answerable with no video at all, and every one of its turn-sequence questions
failed. Read all of this.

A wrong option must be wrong ONLY because the video says otherwise -- never because a reader
can rule it out from the setting. Someone who knows this is a person on foot outdoors, and
nothing else, must find all eight equally possible. So:

- Build them from THIS stretch's own specifics: the landmarks, objects and turns the
  annotation names. Use the identifying detail -- "the red and white striped pole", not
  "a pole" -- in wrong options as much as in right ones. Generic wrong options next to
  specific correct ones is the single clearest tell.
- Make them wrong by COMPOSITION, not by content: the right things in the wrong order, an
  event placed in the wrong phase, two landmarks swapped, a turn given the wrong direction
  or the wrong sharpness, an object moved to the other side of the path.
- Do NOT write each wrong option as a correct one with a single detail edited. If every
  wrong option is one edit from a right one, then on each detail the correct value is the
  majority value, and the answer can be recovered by taking the most common colour, count
  or side -- no video needed. A set built that way once drove blind guessing from 48.6%
  to 75%.
- Vary which details differ and how many: change two or three at once, in different
  combinations, so no option sits at the centre of the set.
- Keep all eight the same length and specificity. The correct options must not be the
  longest or the most detailed -- picking the longest option alone wins 42.2% of
  single-answer questions in an earlier corpus.

Every question carries evidence_spans in seconds of the FULL recording, inside {p0}-{p1}.

Return only:
{{"questions": [{{"question": "...", "question_type": "...",
  "options": ["correct 1", "correct 2", "correct 3", "wrong 1", "wrong 2", "wrong 3", "wrong 4", "wrong 5"],
  "evidence_spans": [{{"start_seconds": 0, "end_seconds": 0, "description": "..."}}],
  "why_hard": "what a viewer must track across the stretch to answer"}}]}}"""

RECHECK_PROMPT = """Verify one benchmark question against the clip, which is exactly the stretch the
question is about.

QUESTION: {question}
OPTIONS: {options}
MARKED CORRECT: {answer}

Watch the clip. Decide whether the marked options -- all of them, and no others -- are the
correct answer.

- "supported"    the marked options are right and the unmarked ones are wrong
- "contradicted" the clip shows something else; say what in notes
- "insufficient" the clip does not settle it

Judge only what the clip shows. If a detail is legible, it counts, whatever the encode.

Return only: {{"verdict": "...", "correct_answer_if_different": null, "notes": "..."}}"""


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - h) / d, 100 * (c + h) / d)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clock_frames(tmpdir, start, length):
    """One PNG per second showing true recording time, for overlay onto the clip."""
    from PIL import Image, ImageDraw, ImageFont
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:
        font = ImageFont.load_default()
    for k in range(int(length) + 2):
        t = int(start) + k
        img = Image.new("RGB", (118, 36), (0, 0, 0))
        ImageDraw.Draw(img).text((8, 4), f"{t // 60:02d}:{t % 60:02d}", fill=(255, 255, 255), font=font)
        img.save(tmpdir / f"tc_{k:05d}.png")
    return tmpdir / "tc_%05d.png"


def cut_with_clock(source, start, length, dst):
    if dst.exists() and 4096 < dst.stat().st_size <= INLINE_CAP:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".tc_{dst.stem}"
    try:
        pattern = clock_frames(tmp, start, length)
        budget = int(INLINE_CAP * 8 * 0.88 / max(length, 1) / 1000)
        for h, kbps in ((480, budget), (432, int(budget * 0.97)), (360, int(budget * 0.95))):
            r = subprocess.run(
                [FFMPEG, "-y", "-ss", str(start), "-t", str(length), "-i", str(source),
                 "-framerate", "1", "-i", str(pattern),
                 "-filter_complex", f"[0:v]scale=-2:{h}[v];[v][1:v]overlay=6:6:shortest=1",
                 "-c:v", "libx264", "-preset", "veryfast", "-b:v", f"{kbps}k",
                 "-maxrate", f"{int(kbps * 1.15)}k", "-bufsize", f"{kbps * 2}k",
                 "-an", "-movflags", "+faststart", str(dst)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size <= INLINE_CAP:
                return "ok"
        return "oversize"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def video_part(path):
    data = Path(path).read_bytes()
    if len(data) > INLINE_CAP:
        raise ValueError(f"clip is {len(data)/2**20:.1f} MiB, over the inline cap")
    return {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(data).decode("ascii")}}


def validate_annotation(d, p0, p1):
    ph = d.get("route_phases")
    if not isinstance(ph, list) or not ph:
        raise ValueError("route_phases required")
    prev = None
    for p in ph:
        a, b = float(p["start_sec"]), float(p["end_sec"])
        if not (p0 - 2 <= a < b <= p1 + 2):
            raise ValueError(f"phase {a}-{b} outside window {p0}-{p1}")
        if prev is not None and abs(a - prev) > 2:
            raise ValueError(f"phases must tile: gap at {prev} -> {a}")
        prev = b
        if not p.get("landmarks"):
            raise ValueError("each phase needs landmarks")
    for k in ("turns", "revisited_landmarks", "environment_stages",
              "dynamic_events", "distinctive_objects"):
        if not isinstance(d.get(k), list):
            raise ValueError(f"{k} must be a list")


def make_q_validator(n, p0, p1):
    def check(d):
        qs = d.get("questions")
        if not isinstance(qs, list) or len(qs) != n:
            raise ValueError(f"need exactly {n} questions, got {len(qs) if isinstance(qs, list) else '?'}")
        for q in qs:
            stem = (q.get("question") or "").strip()
            if not stem:
                raise ValueError("empty question")
            low = stem.lower()
            for bad in LEAK:
                if bad in low:
                    raise ValueError(f"question leaks {bad!r}: {stem[:60]}")
            if q.get("question_type") not in TYPES:
                raise ValueError(f"bad question_type {q.get('question_type')!r}")
            opts = q.get("options")
            if not isinstance(opts, list) or len(opts) != 8:
                raise ValueError(f"need exactly 8 options, got {len(opts) if isinstance(opts, list) else '?'}")
            seen = set()
            for o in opts:
                if not isinstance(o, str) or not o.strip():
                    raise ValueError("options must be non-empty strings")
                if "\n" in o:
                    raise ValueError("one option per string; this one contains a newline")
                key = o.strip().lower()
                if key in seen:
                    raise ValueError(f"duplicate option: {o[:40]!r}")
                seen.add(key)
                for bad in OUT_OF_SCENE:
                    if bad in key:
                        raise ValueError(f"option names out-of-scene {bad!r}: {o[:50]}")
            lengths = [len(o) for o in opts]
            if sorted(range(8), key=lambda i: -lengths[i])[:3] == [0, 1, 2]:
                raise ValueError("the three correct options are the three longest")
            spans = q.get("evidence_spans")
            if not isinstance(spans, list) or not spans:
                raise ValueError("evidence_spans required")
            for s in spans:
                a, b = s.get("start_seconds"), s.get("end_seconds")
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or b <= a:
                    raise ValueError(f"bad span {s}")
                if b < p0 - 5 or a > p1 + 5:
                    raise ValueError(f"span {a}-{b} outside window {p0}-{p1}")
    return check


def validate_verdict(d):
    if d.get("verdict") not in ("supported", "contradicted", "insufficient"):
        raise ValueError(f"bad verdict {d.get('verdict')!r}")


def shuffle(q):
    texts = q["options"]
    correct = set(range(3))
    order = list(range(len(texts)))
    random.Random(hashlib.sha256(q["question"].encode("utf-8")).hexdigest()).shuffle(order)
    q["options"] = {chr(ord("A") + i): texts[s] for i, s in enumerate(order)}
    q["answer"] = sorted(chr(ord("A") + i) for i, s in enumerate(order) if s in correct)
    q["option_order"] = "shuffled"


def collect():
    rows = []
    for path in sorted(glob.glob(f"{SRC}/**/*.json", recursive=True)):
        rec = json.load(open(path))
        source = rec.get("original_video_path") or ""
        if not source or not os.path.exists(source):
            continue
        stem = os.path.basename(source).rsplit(".", 1)[0]
        idx = int(rec.get("chunk_index", 0))
        p0 = float(rec["start_time_sec"])
        p1 = float(rec["end_time_sec"])
        rows.append({"video": stem, "chunk": idx, "source": source, "p0": p0, "p1": p1,
                     "clip": CLIPS / stem / f"{stem}_w{idx}.clock.mp4"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-only", action="store_true")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-per-window", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=str(OUT / "questions.json"))
    a = ap.parse_args()

    rows = collect()
    print(f"{len(rows)} windows from {len({r['video'] for r in rows})} recordings", flush=True)

    if a.clips_only:
        tally = collections.Counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(cut_with_clock, r["source"], r["p0"], r["p1"] - r["p0"], r["clip"]): r
                    for r in rows}
            for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
                tally[f.result()] += 1
                if i % 20 == 0 or i == len(rows):
                    size = sum(p.stat().st_size for p in CLIPS.rglob("*.mp4")) / 2**30
                    print(f"  {i}/{len(rows)} {dict(tally)} {size:.2f} GiB", flush=True)
        print(f"done: {dict(tally)} -> {CLIPS}")
        return 0

    if not (a.pilot or a.all):
        ap.error("pass --clips-only, --pilot N, or --all")

    rows = [r for r in rows if r["clip"].exists()]
    print(f"{len(rows)} windows have a clock clip", flush=True)
    if a.pilot:
        rng = random.Random(SEED)
        by_video = collections.defaultdict(list)
        for r in rows:
            by_video[r["video"]].append(r)
        picked = []
        for v in sorted(by_video):
            g = by_video[v]
            rng.shuffle(g)
            picked.append(g[0])
        rng.shuffle(picked)
        rows = picked[:a.pilot]
        print(f"pilot: {len(rows)} windows", flush=True)

    fv = load("fv", BASE / "build_fullvideo_annotations.py")
    gen = load("gen", BASE / "repair_runningbench_annotations.py")
    api = gen.Floodgate("")
    api.session = fv.PacedSession(api.session, fv.RateLimiter(QUOTA_PER_SECOND))
    OUT.mkdir(parents=True, exist_ok=True)

    questions = []
    done = set()
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out))
            questions = prev.get("questions", [])
            done = {(q["video"], q["chunk"]) for q in questions}
            print(f"resuming: {len(questions)} questions over {len(done)} windows", flush=True)
        except Exception:
            pass
    rows = [r for r in rows if (r["video"], r["chunk"]) not in done]
    print(f"to do: {len(rows)}", flush=True)

    lock = threading.Lock()
    counter = {"n": 0}

    def work(r):
        length = r["p1"] - r["p0"]
        try:
            parts = [{"text": ANNOTATE_PROMPT.format(length=int(length), vid=r["video"],
                                                     p0=int(r["p0"]), p1=int(r["p1"]))},
                     video_part(r["clip"])]
            ann = gen.clean_json(api.generate(gen.CAPTION_MODEL, parts, 16384, json_mode=True))
            validate_annotation(ann, r["p0"], r["p1"])
        except Exception as exc:
            return [], None, f"annotate: {type(exc).__name__}: {exc}"

        try:
            data = gen.generate_valid_json(
                api,
                Q_PROMPT.format(n=a.n_per_window, length=int(length), vid=r["video"],
                                p0=int(r["p0"]), p1=int(r["p1"]),
                                annotation=json.dumps(ann, ensure_ascii=False),
                                types=", ".join(TYPES)),
                make_q_validator(a.n_per_window, r["p0"], r["p1"]), 32768)
        except Exception as exc:
            return [], ann, f"questions: {type(exc).__name__}: {exc}"

        out = []
        for q in data["questions"]:
            shuffle(q)
            q.update({"video": r["video"], "chunk": r["chunk"],
                      "window_sec": [r["p0"], r["p1"]], "clip": str(r["clip"])})
            gates = {"structure": "pass", "leak_scan": "pass"}
            arity = "Exactly three options are correct."
            hits, guesses = 0, []
            for t in range(3):
                try:
                    bg = gen.generate_valid_json(
                        api, fv.BLIND_PROMPT.format(question=q["question"],
                                                   options=json.dumps(q["options"], ensure_ascii=False),
                                                   arity=f"{arity} (attempt {t+1})"),
                        fv.validate_blind, 4096)
                    g = sorted(bg["answer"])
                except Exception:
                    g = ["?"]
                guesses.append(g)
                hits += g == sorted(q["answer"])
            gates["blind_guess"] = {"guesses": guesses, "hits_of_3": hits, "matches_gold": hits >= 2}
            if gates["blind_guess"]["matches_gold"]:
                gates["visual_recheck"] = {"verdict": "skipped", "notes": "blind-guessable"}
            else:
                try:
                    parts = [{"text": RECHECK_PROMPT.format(
                                  question=q["question"],
                                  options=json.dumps(q["options"], ensure_ascii=False),
                                  answer=json.dumps(q["answer"]))},
                             video_part(r["clip"])]
                    gates["visual_recheck"] = gen.clean_json(
                        api.generate(gen.CAPTION_MODEL, parts, 8192, json_mode=True))
                    validate_verdict(gates["visual_recheck"])
                except Exception as exc:
                    gates["visual_recheck"] = {"verdict": "error", "notes": type(exc).__name__}
            q["gates"] = gates
            out.append(q)
        return out, ann, None

    anns = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for got, ann, err in pool.map(work, rows):
            with lock:
                counter["n"] += 1
                i = counter["n"]
                if err:
                    print(f"  !! window {i}: {err}", flush=True)
                if ann:
                    anns[i] = ann
                questions += got
                usable = sum(1 for q in questions
                             if q["gates"].get("visual_recheck", {}).get("verdict") == "supported"
                             and not q["gates"]["blind_guess"].get("matches_gold"))
                blind = sum(1 for q in questions if q["gates"]["blind_guess"].get("matches_gold"))
                print(f"  {i}/{len(rows)} · {len(questions)} q · {usable} usable · {blind} blind",
                      flush=True)
                json.dump({"n": len(questions), "questions": questions},
                          open(a.out, "w"), ensure_ascii=False, indent=1)

    total = len(questions)
    if not total:
        print("no questions produced")
        return 1
    usable = [q for q in questions
              if q["gates"].get("visual_recheck", {}).get("verdict") == "supported"
              and not q["gates"]["blind_guess"].get("matches_gold")]
    blind = sum(1 for q in questions if q["gates"]["blind_guess"].get("matches_gold"))
    lo, hi = wilson(len(usable), total)
    bl, bh = wilson(blind, total)
    print(f"\n=== {total} questions -> {a.out}")
    print(f"usable        {len(usable)}/{total} = {100*len(usable)/total:.1f}%  95% CI [{lo:.1f}, {hi:.1f}]")
    print(f"blind-guessed {blind}/{total} = {100*blind/total:.1f}%  95% CI [{bl:.1f}, {bh:.1f}]")
    print(f"  aggregated-annotation attempt: 68.4% blind [57.3, 77.8]")
    print(f"  segment corpus 48.6% · full-video 8-option multi 7.3%")
    v = collections.Counter()
    for q in questions:
        rc = q["gates"].get("visual_recheck", {})
        v[rc.get("verdict") if isinstance(rc, dict) else rc] += 1
    print(f"recheck       {dict(v)}")
    by_t = collections.defaultdict(collections.Counter)
    for q in questions:
        by_t[q["question_type"]]["ok" if q in usable else "no"] += 1
    for t, c in sorted(by_t.items()):
        n = c["ok"] + c["no"]
        print(f"  {t:26s} {c['ok']:3d}/{n:3d}  {100*c['ok']/n:3.0f}%   (aggregated: "
              f"{'0%' if t == 'window-turn-sequence' else '11-63%'})")
    if total < 100:
        print(f"\nNOTE: n={total}. Read the intervals, not the point estimates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
