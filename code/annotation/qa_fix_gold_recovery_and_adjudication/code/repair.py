#!/usr/bin/env python3
"""Repair the questions the recheck found broken, with the footage in the loop.

Three kinds of damage, three treatments:

  REKEY                   the answer key points at the wrong letter. Both independent
                          viewers agree on another letter -> just move the key. No call.
  BROKEN_NO_CORRECT_OPTION / GOLD_CONTRADICTED_NO_REPLACEMENT
                          nothing in the option list is true of the footage. The stem is
                          usually fine, so rewrite the OPTIONS: Gemini watches the clips,
                          writes the n true statements, then writes distractors out of the
                          video's own material.
  blind_guessable         the footage is irrelevant because the answer is guessable from
                          the option list alone. Rewrite the distractors for prior
                          plausibility ("common but not present", never "uncommon and not
                          present") and iterate adversarially against the blind gate.

Every repair is then re-verified the same way the corpus was audited -- a fresh video
review that never sees the proposed key, plus a fresh blind gate -- and a repair that
fails verification is discarded, leaving the original marked unfixable. A rewrite whose
only evidence is that it was attempted is worse than an honest reject.
"""
import argparse, base64, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floodgate import Floodgate, RateLimiter, CAPTION_MODEL, STRUCTURE_MODEL, clean_json
from video_review import review as video_review_call
from blind_gate import one_vote as blind_vote

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
LETTERS = "ABCDEFGHIJ"
# the key is trustworthy in these -- a guessable one needs new distractors, not new options
KEY_IS_SOUND = {"REKEY", "KEY_FROM_AGREEMENT", "KEY_FROM_GEMINI2", "REKEY_GEMINI_ONLY"}

REWRITE_OPTIONS = """You are repairing a broken multiple-choice question about this egocentric
walking/running footage. You can see the actual clips.

The question below is broken: an independent review of the footage found that NONE of the
current options is fully true (details of what was wrong are given). The stem is fine --
keep the question exactly as it is. Rewrite the OPTION LIST.

QUESTION (do not change it): {question}
It must have exactly {n} correct option(s) and {n_opt} options in total.

CURRENT OPTIONS AND WHAT THE FOOTAGE SAID ABOUT EACH:
{diagnosis}

RULES -- these come from failures, not from theory:
1. Write exactly {n} correct option(s). Each must be true of the footage in EVERY detail,
   and you must give the clip label and MM:SS that shows it.
2. Write {n_wrong} distractors. Each one must be contradicted by the footage.
3. Prior plausibility: a person who has NOT seen the video must find every option about
   equally likely. Build distractors out of things that are COMMON but NOT PRESENT here --
   not things that are uncommon and absent. A distractor nobody would believe anyway makes
   the correct option the only credible one and the question answerable blind.
4. Build distractors from this video's own material: take a landmark/turn/phase that IS in
   the footage and put it in the wrong order, on the wrong side, or in the wrong phase.
   Never invent a setting or object the footage has no trace of (no beach, no motorway, no
   forest, no indoor track, no bridge, no boardwalk unless it is really there).
5. Keep all options the same length, register and level of detail as each other. Length,
   hedging and specificity are all answer-leaking cues.
6. If the route is an out-and-back, say explicitly whether you mean the outbound or the
   return leg, and keep all evidence on one side of the turnaround: outbound left is
   return right.

Return ONLY JSON:
{{"options": {{"A": "...", ... exactly {n_opt} letters ...}},
  "answer": <a list of EXACTLY {n} letter(s), e.g. {answer_example}>,
  "evidence": {{"<each correct letter>": "CLIP_x MM:SS - what is visible"}},
  "distractor_basis": {{"<each wrong letter>": "which real element of this video it twists, and why the footage rules it out"}}}}"""

REWRITE_DISTRACTORS = """This multiple-choice question is answerable WITHOUT watching the video --
a model with no footage picked the correct answer. The correct answer and the stem are
right; the distractors are the problem. You can see the footage.

QUESTION: {question}
CORRECT ANSWER (keep its content, you may only re-letter it): {answer_text}
CURRENT OPTIONS:
{options}
{blind_note}

The failure is always the same shape: the distractors describe things that are unusual, so
the correct option is the only one that sounds like real life. Fix that.
- Replace every wrong option with something COMMON but NOT PRESENT in this footage, or with
  this video's own material twisted (right/left mirrored, two landmarks swapped in order, a
  real event moved to the wrong phase of the route).
- Never introduce a setting or object the footage has no trace of.
- Match the correct option in length, register and specificity.
- Every distractor must still be contradicted by the footage; say how.

Return ONLY JSON:
{{"options": {{"A": "...", ... {n_opt} letters ...}},
  "answer": <a list of EXACTLY {n} letter(s), e.g. {answer_example}>,
  "distractor_basis": {{"<each wrong letter>": "what it twists and how the footage rules it out"}}}}"""


def rewrite_schema(n_opt, n):
    """`evidence` and `distractor_basis` must declare their properties.

    They used to be bare {"type": "object"}. A property-less object in a responseSchema
    comes back as {} -- the model has no declared field to fill -- so every rewrite then
    failed the "evidence missing for correct option(s)" check even though its options and
    answer were fine: 127 of the 127 non-429 failures on 2026-09-16. Declaring one string
    property per letter is what makes the field fillable.
    """
    letters = list(LETTERS[:n_opt])
    per_letter = lambda: {"type": "object",
                          "properties": {l: {"type": "string"} for l in letters}}
    return {"type": "object",
            "properties": {
                "options": {"type": "object",
                            "properties": {l: {"type": "string"} for l in letters},
                            "required": letters},
                "answer": {"type": "array", "items": {"type": "string", "enum": letters},
                           "minItems": n, "maxItems": n},
                "evidence": per_letter(),
                "distractor_basis": per_letter()},
            "required": ["options", "answer", "evidence"]}


def normalize_evidence(raw, letters):
    """Accept every shape the model actually returns and key it by letter.

    Seen in practice: the declared dict {"A": "CLIP_1 00:12 - ..."}; a list of records
    [{"option": "A", "where": "CLIP_1 00:12", "what": "..."}]; and letters carrying
    punctuation or case ("a", "A.", "Option A"). Anything unrecognised is dropped rather
    than guessed at -- a wrong letter here would attach evidence to the wrong option.
    """
    out = {}
    ok = set(letters)

    def put(letter, text):
        if not isinstance(letter, str) or not isinstance(text, str):
            return
        # a standalone letter first, so "Option C" is C and not O
        m = re.search(r"\b([A-Za-z])\b", letter) or re.search(r"([A-Za-z])", letter)
        if not m:
            return
        l = m.group(1).upper()
        if l in ok and text.strip():
            out.setdefault(l, text.strip())

    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, str):
                put(k, v)
            elif isinstance(v, dict):
                put(k, " ".join(str(x) for x in v.values() if x))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            letter = item.get("option") or item.get("letter") or item.get("key")
            text = " ".join(str(item.get(f, "")) for f in ("clip", "where", "timestamp", "what", "evidence", "text")
                            if item.get(f))
            put(letter, text or json.dumps(item, ensure_ascii=False))
    return out


def video_parts(prep):
    parts = []
    for c in prep["clips"]:
        parts.append({"text": f"--- {c['label']} ---"})
        parts.append({"inlineData": {"mimeType": "video/mp4",
                                     "data": base64.b64encode(open(c["file"], "rb").read()).decode("ascii")}})
    return parts


def validate_optionset(d, n_opt, n):
    o = d.get("options")
    if not isinstance(o, dict) or len(o) != n_opt or set(o) != set(LETTERS[:n_opt]):
        return f"options must be exactly {LETTERS[:n_opt]}"
    if any(not isinstance(v, str) or not v.strip() for v in o.values()):
        return "empty option text"
    a = d.get("answer")
    if not isinstance(a, list) or len(set(a)) != n or not set(a) <= set(o):
        return f"answer must be {n} letters from the option set"
    return None


def rewrite_options(api, rec, prep, vrev, writer=CAPTION_MODEL):
    n, n_opt = rec["n_select"], len(rec["options"])
    diag = "\n".join(
        f"{l}. {rec['options'][l]}\n   -> footage says: {vrev['per_option'].get(l,'?')} "
        f"({vrev['evidence'].get(l,'')[:220]})" for l in sorted(rec["options"]))
    prompt = REWRITE_OPTIONS.format(question=rec["question"], n=n, n_opt=n_opt,
                                    n_wrong=n_opt - n, diagnosis=diag,
                                    answer_example=json.dumps(list(LETTERS[:n])))
    last = None
    for _ in range(3):
        try:
            raw = api.generate(writer, [{"text": prompt}] + video_parts(prep),
                               max_tokens=16384, timeout=900, attempts=3,
                               schema=rewrite_schema(n_opt, n))
        except Exception as exc:
            last = str(exc)[:200]; continue
        d = clean_json(raw)
        if not d:
            last = "unparseable"; continue
        err = validate_optionset(d, n_opt, n)
        if not err:
            d["evidence"] = normalize_evidence(d.get("evidence"), LETTERS[:n_opt])
            missing = [l for l in d["answer"] if not d["evidence"].get(l)]
            if missing:
                err = (f"evidence missing for correct option(s) {missing}: "
                       "give CLIP_x MM:SS for each")
        if err:
            last = err
            prompt += f"\n\nYour previous attempt was rejected: {err}. Fix exactly that."
            continue
        return d
    raise RuntimeError(last or "rewrite failed")


def rewrite_distractors(api, rec, prep, answer_text, blind_pick, writer=CAPTION_MODEL):
    n, n_opt = rec["n_select"], len(rec["options"])
    note = (f"\nA model with no video just answered {blind_pick} -- target that specific "
            f"reason it looked right." if blind_pick else "")
    prompt = REWRITE_DISTRACTORS.format(question=rec["question"], answer_text=answer_text,
                                        options="\n".join(f"{l}. {rec['options'][l]}" for l in sorted(rec["options"])),
                                        n=n, n_opt=n_opt, blind_note=note,
                                        answer_example=json.dumps(list(LETTERS[:n])))
    last = None
    for _ in range(3):
        try:
            raw = api.generate(writer, [{"text": prompt}] + video_parts(prep),
                               max_tokens=16384, timeout=900, attempts=3,
                               schema=rewrite_schema(n_opt, n))
        except Exception as exc:
            last = str(exc)[:200]; continue
        d = clean_json(raw)
        if not d:
            last = "unparseable"; continue
        err = validate_optionset(d, n_opt, n)
        if err:
            last = err
            prompt += f"\n\nYour previous attempt was rejected: {err}. Fix exactly that."
            continue
        d["evidence"] = normalize_evidence(d.get("evidence"), LETTERS[:n_opt])
        d["distractor_basis"] = normalize_evidence(d.get("distractor_basis"), LETTERS[:n_opt])
        return d
    raise RuntimeError(last or "rewrite failed")


def verify(api, rec, prep, cand, verifier=CAPTION_MODEL, blind_model=STRUCTURE_MODEL):
    """Fresh eyes on the repaired question: a video review that never sees the proposed
    key, then a blind gate. Both must pass.

    `verifier` should not be the model that wrote the rewrite. With both set to
    CAPTION_MODEL -- the default this pipeline ran with until 2026-09-16 -- the model
    that invented the options is also the one certifying them, which is the
    self-confirmation the rest of this audit is built to avoid.
    """
    probe = dict(rec, options=cand["options"], gold=cand["answer"])
    vr = video_review_call(api, probe, prep, vote=7, model=verifier)
    if vr.get("error"):
        return {"ok": False, "why": "verify video: " + vr["error"]}
    want = set(cand["answer"])
    if set(vr["final_answer"]) != want:
        return {"ok": False, "why": f"video review answered {vr['final_answer']}, key says {sorted(want)}",
                "video": vr}
    if vr["verdict"] != "clear":
        return {"ok": False, "why": f"video review verdict {vr['verdict']}", "video": vr}
    hits = 0
    votes = []
    for v in range(3):
        try:
            pick = blind_vote(api, probe, v, blind_model)
        except Exception as exc:
            return {"ok": False, "why": "verify blind: " + str(exc)[:120], "video": vr}
        votes.append(pick)
        hits += set(pick) == want
    if hits >= 2:
        return {"ok": False, "why": f"still blind-guessable ({hits}/3)", "video": vr, "blind": votes}
    return {"ok": True, "video": vr, "blind": votes, "blind_hits": hits}


def repair_one(api, rec, prep, adj, vrev, rounds=3, writer=CAPTION_MODEL,
               verifier=CAPTION_MODEL, blind_model=STRUCTURE_MODEL):
    rid = rec["review_id"]
    out = {"id": rid, "review_id": rid, "decision": adj["decision"], "attempts": [],
           "writer_model": writer, "verify_model": verifier}
    sound = adj["decision"].startswith("KEEP") or adj["decision"] in KEY_IS_SOUND
    mode = "distractors" if sound and adj.get("guessable") else "options"
    cur = dict(rec)
    blind_pick = None
    for rnd in range(rounds):
        try:
            if mode == "options":
                cand = rewrite_options(api, cur, prep, vrev, writer)
            else:
                key = adj.get("new_gold") or adj["gold"]
                ans_text = " / ".join(rec["options"][l] for l in key) if key else ""
                cand = rewrite_distractors(api, cur, prep, ans_text, blind_pick, writer)
        except Exception as exc:
            out["attempts"].append({"round": rnd, "error": str(exc)[:200]})
            continue
        res = verify(api, dict(rec, options=cand["options"]), prep, cand, verifier, blind_model)
        out["attempts"].append({"round": rnd, "options": cand["options"], "answer": cand["answer"],
                                "ok": res["ok"], "why": res.get("why"),
                                "blind": res.get("blind"), "blind_hits": res.get("blind_hits")})
        if res["ok"]:
            out["repaired"] = True
            out["options"] = cand["options"]
            out["answer"] = cand["answer"]
            out["evidence"] = cand.get("evidence") or {}
            out["distractor_basis"] = cand.get("distractor_basis") or {}
            out["rounds_used"] = rnd + 1
            return out
        if res.get("blind"):
            blind_pick = res["blind"][0]
        cur = dict(rec, options=cand["options"])
    out["repaired"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default="BROKEN_NO_CORRECT_OPTION,GOLD_CONTRADICTED_NO_REPLACEMENT")
    ap.add_argument("--guessable-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rps", type=float, default=0.45)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", help="file of review_ids, one per line")
    ap.add_argument("--units", help="comma-separated units, e.g. cross_video,cross_recording")
    ap.add_argument("--min-clips", type=int, default=0, help="only questions with >= this many clips")
    ap.add_argument("--writer-model", default=CAPTION_MODEL)
    ap.add_argument("--verify-model", default=CAPTION_MODEL,
                    help="set this to a DIFFERENT model than --writer-model")
    ap.add_argument("--blind-model", default=STRUCTURE_MODEL)
    ap.add_argument("--out", default=f"{QA}/repairs.jsonl")
    a = ap.parse_args()

    idx = {json.loads(l)["review_id"]: json.loads(l) for l in open(f"{QA}/master_index.jsonl")}
    prep = {}
    for l in open(f"{QA}/clipprep.jsonl"):
        r = json.loads(l); prep[r["review_id"]] = r
    vrev = {}
    for l in open(f"{QA}/video_review.jsonl"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if not r.get("error") and r.get("vote") == 0:
            vrev[r["review_id"]] = r
    adjs = {json.loads(l)["review_id"]: json.loads(l) for l in open(f"{QA}/adjudication.jsonl")}
    want = set(a.decisions.split(",")) if a.decisions else set()
    todo = []
    for rid, adj in adjs.items():
        if a.guessable_only:
            if not adj.get("guessable"):
                continue
        elif adj["decision"] not in want:
            continue
        if rid in vrev and rid in prep:
            todo.append(rid)
    if a.only:
        keep = {x.strip() for x in open(a.only) if x.strip()}
        todo = [r for r in todo if r in keep]
    if a.units:
        units = set(a.units.split(","))
        todo = [r for r in todo if idx[r]["unit"] in units]
    if a.min_clips:
        todo = [r for r in todo if len(idx[r]["clips"]) >= a.min_clips]
    # The bundle media was lost with the machine that built it and restored from HF; a
    # question whose clips are not on disk would crash the worker on open(), taking the
    # whole batch's unflushed futures with it. Skip and report instead.
    absent = [r for r in todo if not all(os.path.exists(c["file"]) for c in prep[r]["clips"])]
    if absent:
        print(f"skipping {len(absent)} questions whose clip files are missing", flush=True)
        todo = [r for r in todo if r not in set(absent)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [r for r in todo if r not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"repairing {len(todo)} questions (mode={'guessable' if a.guessable_only else a.decisions}) "
          f"writer={a.writer_model} verifier={a.verify_model} blind={a.blind_model}", flush=True)

    api = Floodgate(limiter=RateLimiter(a.rps))
    lock = threading.Lock()
    n = fixed = 0
    with open(a.out, "a") as fh, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(repair_one, api, idx[r], prep[r], adjs[r], vrev[r], a.rounds,
                            a.writer_model, a.verify_model, a.blind_model) for r in todo]
        for f in as_completed(futs):
            r = f.result()
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
                n += 1; fixed += bool(r.get("repaired"))
                if n % 10 == 0:
                    print(f"{n}/{len(todo)} repaired={fixed}", flush=True)
    print(f"DONE {n} repaired={fixed}", flush=True)


if __name__ == "__main__":
    main()
