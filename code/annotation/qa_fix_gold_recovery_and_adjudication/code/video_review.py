#!/usr/bin/env python3
"""Gate 5 (high-res visual recheck) re-run over the whole HF crowdsourcing corpus.

For every one of the 1,526 questions: show the real footage to gemini-3.1-pro-preview
and make it do exactly what the human annotators were asked to do -- judge EVERY
option against the pixels, cite the second, then answer. The gold answer is never
shown, so the result is an independent vote that can be compared against both the
recovered gold and the human submissions.

Option letters are re-shuffled per vote (deterministically, by vote index) and mapped
back afterwards, so a second vote is not just the first vote re-read and position
bias cannot masquerade as agreement.
"""
import argparse, base64, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floodgate import (Floodgate, RateLimiter, CAPTION_MODEL, clean_json)

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
MAX_INLINE = 20 * 1024 * 1024          # hard cap from recheck_segments_60s.py
MAX_VIDEO_FILES = 10                   # hard cap from the Gemini serving layer

PROMPT = """You are verifying a multiple-choice question about egocentric walking/running
footage. You can see the actual footage. Do NOT use general plausibility or world
knowledge -- judge ONLY what these clips show.

CLIP LABELS
- CLIP_1, CLIP_2, ... are time-ordered cuts from ONE recording (CLIP_1 is earliest).
- CLIP_A, CLIP_B, ... are anonymous clips referred to by letter in the question; their
  letter order says NOTHING about time.
- VIDEO_A, VIDEO_B, ... are anonymous whole recordings used by cross-recording questions.
Each clip is given to you below, labelled, in the order listed.

PROCEDURE (follow it literally, do not judge on overall impression)
1. Watch every clip end to end and build a timeline per clip: turns (and which way),
   surface changes, anything with a number/text/distinctive colour, moving people and
   vehicles, prominent buildings.
2. Then go through the options ONE AT A TIME. For each option decide:
   - "supported": you can point at the clip and second that shows it.
   - "ruled_out": the footage contradicts it. ONE wrong detail makes the whole option
     wrong (side swapped, order reversed, colour wrong, wrong phase of the route).
     If the footage does cover that place/moment and the thing is not there, that is
     also "ruled_out".
   - "undecidable": the footage never covers the place or the moment the option talks
     about, or it is too blurred/dark to tell.
3. Give the evidence for every option: the clip label and a MM:SS time inside that clip
   (times are within the clip, it starts at 00:00), plus what is visible. For
   "undecidable" say what is missing instead.
4. final_answer: pick exactly {n} option letter(s) -- the ones best supported. If fewer
   than {n} options are supported, still list your {n} best and say so in notes.
5. verdict:
   - "clear": exactly one defensible answer set, visible on screen.
   - "ambiguous": a second set of options is also defensible, or the question stem is
     ambiguous, or the number of supported options is not {n}.
   - "cant_tell": the footage is missing/blurred, or never covers the key moment.

QUESTION: {question}

OPTIONS (this question wants exactly {n} of them):
{options}

Return ONLY JSON:
{{"per_option": {{"A": "supported|ruled_out|undecidable", ...for every letter...}},
  "evidence": {{"A": "CLIP_x MM:SS - what is visible", ...for every letter...}},
  "final_answer": ["..."], "verdict": "clear|ambiguous|cant_tell",
  "notes": "which option was the problem and why, <=400 chars"}}"""


def schema_for(letters):
    enum = ["supported", "ruled_out", "undecidable"]
    return {"type": "object",
            "properties": {
                "per_option": {"type": "object",
                               "properties": {l: {"type": "string", "enum": enum} for l in letters},
                               "required": letters},
                "evidence": {"type": "object",
                             "properties": {l: {"type": "string"} for l in letters}},
                "final_answer": {"type": "array", "items": {"type": "string", "enum": letters}},
                "verdict": {"type": "string", "enum": ["clear", "ambiguous", "cant_tell"]},
                "notes": {"type": "string"}},
            "required": ["per_option", "final_answer", "verdict"]}


def shuffled_view(options, vote):
    """Return (display_options, disp->true letter map). vote 0 keeps the original order
    (so it is comparable with the human pass, which saw sorted keys)."""
    true_letters = sorted(options)
    if vote == 0:
        return {l: options[l] for l in true_letters}, {l: l for l in true_letters}
    order = true_letters[:]
    random.Random(f"{vote}").shuffle(order)
    disp = {}
    back = {}
    for i, t in enumerate(order):
        d = true_letters[i]
        disp[d] = options[t]
        back[d] = t
    return disp, back


def review(api, rec, prep, vote, model=CAPTION_MODEL):
    rid = rec["review_id"]
    if prep.get("error"):
        return {"id": f"{rid}#{vote}", "review_id": rid, "vote": vote, "error": "clipprep: " + prep["error"]}
    files = prep["clips"]
    # Every Gemini model on Floodgate refuses more than 10 video files per request. Say so
    # here: as a bare 400 from the API it looks like a malformed request and the three
    # questions it hit sat in the corpus as "not_reviewed". merge_overlimit_clips.py
    # collapses same-label excerpts to get under the cap.
    if len(files) > MAX_VIDEO_FILES:
        return {"id": f"{rid}#{vote}", "review_id": rid, "vote": vote,
                "error": f"{len(files)} video files > model cap of {MAX_VIDEO_FILES}; "
                         "run merge_overlimit_clips.py"}
    payload = sum(int(b["bytes"] * 4 / 3) + 200 for b in files)
    if payload > MAX_INLINE:
        return {"id": f"{rid}#{vote}", "review_id": rid, "vote": vote,
                "error": f"inline payload too large ({payload/1048576:.1f} MB b64)"}
    disp, back = shuffled_view(rec["options"], vote)
    letters = sorted(disp)
    opt_text = "\n".join(f"{l}. {disp[l]}" for l in letters)
    parts = [{"text": PROMPT.format(n=rec["n_select"], question=rec["question"], options=opt_text)}]
    for f in files:
        parts.append({"text": f"--- {f['label']} ---"})
        parts.append({"inlineData": {"mimeType": "video/mp4",
                                     "data": base64.b64encode(open(f["file"], "rb").read()).decode("ascii")}})

    def validate(d):
        po = d.get("per_option")
        if not isinstance(po, dict) or set(po) != set(letters):
            return f"per_option keys {sorted(po) if isinstance(po, dict) else po} != {letters}"
        if not isinstance(d.get("final_answer"), list):
            return "final_answer not a list"
        return None

    last = None
    for _ in range(3):
        try:
            raw = api.generate(model, parts, max_tokens=16384, timeout=900, attempts=3,
                               schema=schema_for(letters))
        except Exception as exc:
            last = f"generate: {str(exc)[:200]}"
            continue
        d = clean_json(raw)
        if d is None:
            last = f"unparseable: {str(raw)[:160]}"
            continue
        err = validate(d)
        if err:
            last = err
            continue
        # map display letters back to the real ones
        po = {back[l]: v for l, v in d["per_option"].items()}
        ev = {back[l]: v for l, v in (d.get("evidence") or {}).items() if l in back}
        fa = sorted({back[l] for l in d["final_answer"] if l in back})
        return {"id": f"{rid}#{vote}", "review_id": rid, "vote": vote, "bundle": rec["bundle"],
                "source": rec["source"], "question_type": rec["question_type"],
                "n_select": rec["n_select"], "model": model,
                "per_option": po, "evidence": ev, "final_answer": fa,
                "verdict": d.get("verdict"), "notes": (d.get("notes") or "")[:600],
                "truncated": bool(d.get("truncated")),
                "payload_mb": round(payload / 1048576, 1)}
    return {"id": f"{rid}#{vote}", "review_id": rid, "vote": vote, "error": last or "no attempts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vote", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="file of review_ids, one per line")
    ap.add_argument("--out", default=f"{QA}/video_review.jsonl")
    ap.add_argument("--model", default=CAPTION_MODEL,
                    help="cross-model check: a different model makes the vote independent "
                         "rather than the same model re-reading its own shuffle")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(f"{QA}/master_index.jsonl")]
    prep = {}
    for l in open(f"{QA}/clipprep.jsonl"):
        r = json.loads(l)
        prep[r["review_id"]] = r
    if a.only:
        keep = {l.strip() for l in open(a.only) if l.strip()}
        recs = [r for r in recs if r["review_id"] in keep]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get("error") and r.get("id"):
                done.add(r["id"])
    todo = [r for r in recs if f"{r['review_id']}#{a.vote}" not in done and r["review_id"] in prep]
    if a.limit:
        todo = todo[: a.limit]
    print(f"vote={a.vote} model={a.model} corpus={len(recs)} already_done={len(done)} todo={len(todo)}", flush=True)

    api = Floodgate(limiter=RateLimiter(a.rps))
    lock = threading.Lock()
    n = ok = 0
    with open(a.out, "a") as fh, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(review, api, r, prep[r["review_id"]], a.vote, a.model) for r in todo]
        for f in as_completed(futs):
            r = f.result()
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
                n += 1; ok += not r.get("error")
                if n % 25 == 0:
                    print(f"{n}/{len(todo)} ok={ok}", flush=True)
    print(f"DONE {n} ok={ok}", flush=True)


if __name__ == "__main__":
    main()
