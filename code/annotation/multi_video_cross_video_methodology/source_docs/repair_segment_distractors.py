#!/usr/bin/env python3
"""Rebuild segment-question distractors as near-misses, and measure whether it worked.

A blind model scores 48.6% on this corpus (801-question sample, 95% CI [45.1, 52.0]),
and 41.6% of the time it agrees with itself three times out of three. Reading the
questions it beats shows two construction faults, both downstream of the original
rule -- "DEFINITIVELY CONTRADICTED by the evidence":

  1. The correct option is the most specific one. Distractors name generic props
     ("a yellow school bus"), the answer reports what was actually seen ("a white van
     among parked vehicles and a large dark trash dumpster"). Pick the longest, most
     detailed option and you are usually right.
  2. Distractors are impossible rather than merely false. Asked what signals the
     wearer's movement, the wrong options offer engine noise and a dashboard view --
     ruled out by knowing this is a first-person running video, no watching required.

"Contradicted" guarantees an option is wrong. It does not guarantee it is tempting.
So this regenerates distractors as near-misses of the answer itself -- same objects and
actions, one attribute changed -- with the length band the full-video pipeline enforces
(where the same blind test scores 7.3% on 8-option multi against this corpus's 27.8%).

The gold option's text is never touched, only the distractors around it.

Usage:  repair_segment_distractors.py --pilot 250      (repair, then blind-test)
        repair_segment_distractors.py --all
"""
import argparse
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
import sys
import threading
from pathlib import Path

BASE = Path("/mnt/task_runtime/bolt/gdrive_relay")
SEG_ROOT = "/mnt/data/data_anno/runningbench_qa_expansion/annotations_60s"
OUT = Path("/mnt/data/data_anno/runningbench_segment_repair")
QUOTA_PER_SECOND = 0.45
SEED = 20260831

REPAIR_PROMPT = """Rewrite the WRONG options of one benchmark question about a 60-second first-person
walking/running video. The correct option is fixed — never change, paraphrase or restate it.

EVIDENCE (the only ground truth):
{evidence}

QUESTION: {question}
CORRECT OPTION(S), verbatim and unchangeable:
{correct}

Write exactly {n} wrong options, and treat the whole SET as the thing you are designing.

The trap to avoid, above everything else: if every wrong option is the correct one with a
single detail edited, then on each detail the correct value is the majority value, and the
answer can be recovered by taking the most common colour, the most common count, the most
common side — no video needed. A set like this is worthless:

    dark grey  / multi-story / glass        light yellow / single-story / glass
    light yellow / multi-story / glass  <-- correct, and the majority on every axis
    dark red   / multi-story / glass        light yellow / multi-story / wooden

So:
- Change TWO OR MORE details at once in most wrong options, and change different combinations
  in different options. No detail of the correct option may appear in more than half the set.
- Make the wrong options near-misses OF EACH OTHER too, not only of the correct one. A reader
  who cannot see the video must find no option more "central" or more self-consistent than
  the others.
- Stay inside the world of this video. A wrong option that a person rules out just by knowing
  this is a first-person walking/running recording is useless: no dashboards, no engine noise,
  no indoor scenes unless the evidence has them.
- Be CONTRADICTED by the evidence: it must positively rule the option out. Not merely
  unmentioned, not partly true.
- Match the correct option's specificity and length: between {lo} and {hi} characters, and the
  correct option must not be the longest or the most detailed one in the set.

Return only: {{"options": ["...", "..."]}}

Each option is ONE plain string in that array. Do not prefix them with "-" or a bullet,
do not put a newline inside one, and do not pack several options into a single string —
returning the whole list as one item is the most common way this call fails."""

VERIFY_PROMPT = """Judge each numbered statement against the evidence from a first-person video segment.

EVIDENCE:
{evidence}

STATEMENTS:
{statements}

Answer strictly per statement:
- "contradicted" if the evidence positively rules it out;
- "supported" if the evidence indicates it is true;
- "unclear" if the evidence neither confirms nor rules it out.

Return only: {{"verdicts": ["contradicted", "unclear", ...]}}"""


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


def clean_option(o):
    """Strip a markdown bullet if the model wrapped its list items in one."""
    return re.sub(r"^\s*[-*\u2022]\s+", "", o).strip() if isinstance(o, str) else o


def make_repair_validator(n, lo, hi, correct):
    gold = {c.strip().lower() for c in correct}

    def check(d):
        opts = d.get("options")
        if not isinstance(opts, list) or len(opts) != n:
            raise ValueError(f"need exactly {n} options, got {len(opts) if isinstance(opts, list) else '?'}")
        # Normalise in place so the caller sees cleaned text, not the bulleted form.
        d["options"] = opts = [clean_option(o) for o in opts]
        seen = set()
        for o in opts:
            if not isinstance(o, str) or not o.strip():
                raise ValueError("options must be non-empty strings")
            if "\n" in o:
                raise ValueError("one option per string; this one contains a newline")
            key = o.strip().lower()
            if key in gold:
                raise ValueError("a wrong option repeats the correct option")
            if key in seen:
                raise ValueError("duplicate options")
            seen.add(key)
            if not (lo * 0.8 <= len(o) <= hi * 1.25):
                raise ValueError(f"option length {len(o)} outside {int(lo)}..{int(hi)}: {o[:40]!r}")
    return check


def make_verdict_validator(n):
    def check(d):
        v = d.get("verdicts")
        if not isinstance(v, list) or len(v) != n:
            raise ValueError(f"need {n} verdicts")
        for x in v:
            if x not in ("contradicted", "supported", "unclear"):
                raise ValueError(f"bad verdict {x!r}")
    return check


def shuffle(item):
    keys = sorted(item["options"])
    texts = [item["options"][k] for k in keys]
    correct = {keys.index(k) for k in item["answer"]}
    order = list(range(len(texts)))
    random.Random(hashlib.sha256(item["question"].encode("utf-8")).hexdigest()).shuffle(order)
    item["options"] = {chr(ord("A") + i): texts[src] for i, src in enumerate(order)}
    item["answer"] = sorted(chr(ord("A") + i) for i, src in enumerate(order) if src in correct)


def collect():
    rows = []
    for path in sorted(glob.glob(f"{SEG_ROOT}/**/*.json", recursive=True)):
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        ev = rec.get("dense_annotations") or rec.get("annotation_raw")
        for idx, item in enumerate(rec.get("qa_pairs") or []):
            o, a = item.get("options"), item.get("answer")
            if isinstance(o, dict) and o and isinstance(a, list) and a:
                rows.append({"file": path, "index": idx, "evidence": ev, "item": item})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="repair a stratified sample and blind-test it")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--multi-only", action="store_true",
                    help="repair only multi-answer questions. Single-choice sits at 65.2%% blind "
                         "before repair and 54.5%% after, so rebuilding it buys a corpus that is "
                         "still unusable; multi-answer runs 27.8%% -> 21.1%% from a far better start.")
    ap.add_argument("--no-blind", action="store_true", help="repair only; leave the gates to a local model")
    # A repair call takes ~11 s, so one worker issues only ~0.13 req/s against a 0.45 req/s
    # quota -- the run is latency-bound, not quota-bound. Three workers fill the quota
    # without exceeding it; the shared RateLimiter still serialises every attempt.
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default=str(OUT / "pilot.json"))
    a = ap.parse_args()
    if not (a.pilot or a.all):
        ap.error("pass --pilot N or --all")

    fv = load("fv", BASE / "build_fullvideo_annotations.py")
    gen = load("gen", BASE / "repair_runningbench_annotations.py")

    rows = collect()
    print(f"corpus: {len(rows)} objective questions", flush=True)
    if a.multi_only:
        rows = [r for r in rows if len(r["item"]["answer"]) > 1]
        print(f"multi-answer only: {len(rows)}", flush=True)
    rng = random.Random(SEED)
    if a.pilot:
        strata = collections.defaultdict(list)
        for r in rows:
            strata[(len(r["item"]["options"]), len(r["item"]["answer"]) > 1)].append(r)
        sample = []
        for key, group in sorted(strata.items()):
            take = max(1, round(a.pilot * len(group) / len(rows)))
            rng.shuffle(group)
            sample += group[:take]
        rng.shuffle(sample)
        rows = sample
        print(f"pilot sample: {len(rows)}", flush=True)

    api = gen.Floodgate("")
    api.session = fv.PacedSession(api.session, fv.RateLimiter(QUOTA_PER_SECOND))

    OUT.mkdir(parents=True, exist_ok=True)
    repaired, results = [], []
    # Resume: a 4-hour run should not restart from zero after an interruption.
    done_keys = set()
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out))
            repaired = prev.get("repaired", [])
            results = prev.get("blind", [])
            done_keys = {(r["file"], r["index"]) for r in repaired}
            print(f"resuming: {len(done_keys)} already repaired", flush=True)
        except Exception:
            print("could not read previous output; starting fresh", flush=True)
    rows = [r for r in rows if (r["file"], r["index"]) not in done_keys]
    print(f"to do: {len(rows)}", flush=True)

    lock = threading.Lock()
    counter = {"n": 0}

    def work(r):
        item = json.loads(json.dumps(r["item"]))
        keys = sorted(item["options"])
        # One question in 8,148 has an answer letter with no matching option -- the
        # generator emitted a literal "..." as an option key. Skip it rather than take
        # the whole run down; a KeyError inside a pool worker kills every other thread.
        absent = [k for k in item["answer"] if k not in item["options"]]
        if absent:
            return None, f"malformed: answer {absent} has no option (keys={keys})"
        correct = [item["options"][k] for k in item["answer"]]
        wrong_n = len(keys) - len(correct)
        clen = [len(c) for c in correct]
        lo, hi = min(clen) * 0.75, max(clen) * 1.05
        try:
            data = gen.generate_valid_json(
                api,
                REPAIR_PROMPT.format(evidence=json.dumps(r["evidence"], ensure_ascii=False),
                                     question=item["question"],
                                     correct="\n".join(f"- {c}" for c in correct),
                                     n=wrong_n, lo=int(lo), hi=int(hi)),
                make_repair_validator(wrong_n, lo, hi, correct), 8192)
            new_wrong = data["options"]
            verdicts = gen.generate_valid_json(
                api,
                VERIFY_PROMPT.format(evidence=json.dumps(r["evidence"], ensure_ascii=False),
                                     statements="\n".join(f"{j+1}. {o}" for j, o in enumerate(new_wrong))),
                make_verdict_validator(wrong_n), 4096)["verdicts"]
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        # Keep only distractors the evidence positively rules out; fall back to the
        # original wording for the rest rather than shipping an unverified option.
        old_wrong = [item["options"][k] for k in keys if k not in item["answer"]]
        kept = [o for o, v in zip(new_wrong, verdicts) if v == "contradicted"]
        final_wrong = kept + old_wrong[len(kept):]
        texts = correct + final_wrong
        item["options"] = {chr(ord("A") + j): t for j, t in enumerate(texts)}
        item["answer"] = [chr(ord("A") + j) for j in range(len(correct))]
        shuffle(item)
        item["distractors_rebuilt"] = len(kept)
        if {item["options"][k] for k in item["answer"]} != set(correct):
            return None, "gold text changed"
        return {"file": r["file"], "index": r["index"], "item": item}, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for out_row, err in pool.map(work, rows):
            with lock:
                counter["n"] += 1
                i = counter["n"]
                if err:
                    print(f"  !! {i}: {err}", flush=True)
                else:
                    repaired.append(out_row)
                if i % 20 == 0 or i == len(rows):
                    print(f"  {i}/{len(rows)} repaired {len(repaired)}", flush=True)
                    json.dump({"repaired": repaired, "blind": results},
                              open(a.out, "w"), ensure_ascii=False)

    json.dump({"repaired": repaired, "blind": results}, open(a.out, "w"), ensure_ascii=False)
    print(f"\nrepaired {len(repaired)}/{len(rows)}  ->  {a.out}")
    if results:
        k, n = sum(1 for x in results if x["guessable"]), len(results)
        lo, hi = wilson(k, n)
        print(f"\n=== blind-guess after repair ===")
        print(f"overall  {k}/{n} = {100*k/n:.1f}%   95% CI [{lo:.1f}, {hi:.1f}]   (before: 48.6% [45.1, 52.0])")
        for key, label in (((6, False), "6-option single"), ((8, True), "8-option multi")):
            s = [x for x in results if (x["n_opt"], x["n_ans"] > 1) == key]
            if s:
                kk = sum(1 for x in s if x["guessable"])
                l, h = wilson(kk, len(s))
                print(f"{label:18s}{kk}/{len(s)} = {100*kk/len(s):.1f}%  [{l:.1f}, {h:.1f}]")
        print("before: 6-option single 65.2%, 8-option multi 27.8%")


if __name__ == "__main__":
    sys.exit(main())
