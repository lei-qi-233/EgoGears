#!/usr/bin/env python3
"""Second-generation distractor repair: balance prior plausibility, and iterate adversarially.

Why v1 stalls at 73.4% blind-guessable on single-choice. Reading the questions it still
loses shows one dominant failure, and it is structural rather than sloppy writing:

    Q: What immediately borders both sides of the paved asphalt path?
       grey gravel / metal fence / tall bushes / GREEN GRASS / brick walls / sandy soil

Every distractor there is correctly contradicted by the evidence. That is exactly the
problem. "Contradicted by the evidence" means "not what is actually there", and what is
actually there is usually the ordinary thing -- so the rule that makes a distractor
verifiably wrong also tends to make it a-priori unusual. The correct option is left as
the only typical-sounding item in the set, and a blind reader takes the base rate and
wins. v1's prompt never asks for prior plausibility to be balanced at all.

So v2 changes two things:

  1. Prior-plausibility parity is now an explicit, first-class requirement: each wrong
     option must be something a reader who has never seen this video would rate AT LEAST
     as likely as the correct one. Typical-but-absent beats unusual-but-absent.
  2. The loop is adversarial. After each rewrite the question is blind-tested, and if the
     model still wins, its actual guess is handed back to the rewriter as the thing to
     defeat. v1 was one-shot and never learned what beat it.

Distractors are still verified as contradicted by the evidence, and a rewrite that fails
that check is regenerated -- prior plausibility must not be bought with distractors that
are merely unverifiable.

Validation, not faith: run with --targets to point this at the questions v1 failed on and
report how many it rescues. The project has been burned once by a rewrite that improved
its target metric while blind-guessability got worse, so nothing here is rolled out on
the strength of the prompt reading well.
"""
import argparse, collections, json, math, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate_segments_60s import BLIND_PROMPT, Floodgate, RateLimiter, parse, wilson
from repair_and_measure import (VERIFY_PROMPT, json_call, shuffle_letters,
                                validate_repair, blind)

QUOTA_PER_SECOND = 0.45
SEED = 20260906

REPAIR_V2 = """Rewrite the WRONG options of one benchmark question about a 60-second first-person
walking/running video. The correct option is fixed — never change, paraphrase or restate it.

EVIDENCE (the only ground truth):
{evidence}

QUESTION: {question}
CORRECT OPTION(S), verbatim and unchangeable:
{correct}

Write exactly {n} wrong options.

THE FAILURE YOU MUST FIX — read this twice.

A wrong option has to be two things at once, and the second one is the one that gets
forgotten:

  (a) CONTRADICTED by the evidence — the evidence must positively rule it out.
  (b) At least as ORDINARY, as a-priori likely, as the correct option — to someone who
      has never seen this video.

Requirement (a) pushes hard against (b): what is contradicted is what is not there, and
what is not there is often the unusual thing. Give in to that and you produce a set like

    grey gravel / metal fence / tall bushes / GREEN GRASS / brick walls / sandy soil

where every wrong option is dutifully contradicted and the answer is still obvious,
because grass is simply the most ordinary thing to find beside a path. The reader never
needed the video. That set is a failure no matter how correct its contradictions are.

So: pick wrong options that are TYPICAL of this kind of scene in general and happen to be
absent from THIS segment. Typical-but-absent, never unusual-but-absent. If the correct
option names the commonplace thing, your wrong options must name other commonplace things
— the ones a person would expect just as readily.

Test your own set before answering: if you had only the question and the options, and had
to bet, would any option stand out as the most natural thing to say? If yes, rewrite it.

Also:
- Change TWO OR MORE details at once in most wrong options, in different combinations. No
  detail of the correct option may appear in more than half the set.
- Make the wrong options near-misses OF EACH OTHER, not only of the correct one.
- Stay inside the world of this video: no dashboards, no engine noise, no indoor scenes
  unless the evidence has them.
- Between {lo} and {hi} characters, and the correct option must not be the longest or the
  most detailed one in the set.
{feedback}
Return only: {{"options": ["...", "..."]}}

Each option is ONE plain string in that array. Do not prefix them with "-" or a bullet, do
not put a newline inside one, and do not pack several options into a single string."""

FEEDBACK_TMPL = """
WHAT ALREADY FAILED — a model that was shown NO video was given your previous option set
and picked the correct answer anyway. Its pick: {guess}
Previous set:
{prev}
Whatever cue let it do that is still there. Do not repeat that set. The correct option is
still recognisable as the most plausible or most natural-sounding item — find what makes
it stand out and give the wrong options that same quality.
"""


def verify_contradicted(api, ev, wrong):
    v = json_call(api, VERIFY_PROMPT.format(
        evidence=json.dumps(ev, ensure_ascii=False),
        statements="\n".join(f"{i+1}. {t}" for i, t in enumerate(wrong))),
        lambda d: (d["verdicts"] if isinstance(d.get("verdicts"), list)
                   and len(d["verdicts"]) == len(wrong)
                   else (_ for _ in ()).throw(ValueError("bad verdicts"))), 4096)
    return v, sum(1 for x in v if x == "contradicted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--targets", required=True,
                    help="repair_measure.progress.jsonl — v2 is aimed at the rows v1 lost")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-contradicted", type=float, default=0.8,
                    help="regenerate if fewer than this fraction of distractors are contradicted")
    ap.add_argument("--out", default="repair_v2.json")
    ap.add_argument("--checkpoint", default="repair_v2.progress.jsonl")
    a = ap.parse_args()

    qs = {q["id"]: q for q in (json.loads(l) for l in open(f"{a.qdir}/questions/segments_60s.jsonl"))}
    caps = {}
    for l in open(f"{a.qdir}/captions/segments_60s.jsonl"):
        c = json.loads(l)
        caps[(c["video"], tuple(c["span_sec"]))] = c["caption"]

    targets = []
    for l in open(a.targets):
        r = json.loads(l)
        if r.get("error") or r.get("arity") != "single":
            continue
        if r.get("control_guessable") and r.get("treat_guessable") and r["id"] in qs:
            targets.append((qs[r["id"]], r))       # v1 tried and lost
    random.Random(SEED).shuffle(targets)
    if a.limit:
        targets = targets[:a.limit]
    print(f"v1 failures to attack: {len(targets)}  rounds={a.rounds}", flush=True)

    done = {}
    cp = Path(a.checkpoint)
    if cp.exists():
        for l in cp.open():
            try:
                r = json.loads(l)
                done[r["id"]] = r
            except Exception:
                pass
        print(f"resuming: {len(done)}", flush=True)

    api = Floodgate(os.environ.get("FLOODGATE_PROJECT_TOKEN", ""), RateLimiter(QUOTA_PER_SECOND))
    fh = cp.open("a")
    lock = threading.Lock()
    todo = [(q, r) for q, r in targets if q["id"] not in done]
    t0, counter = time.time(), {"n": 0}
    need = max(1, (a.votes + 1) // 2)

    def work(pair):
        q, v1 = pair
        rec = {"id": q["id"], "n_options": q.get("n_options"), "rounds_used": 0}
        try:
            opts, ans = q["options"], sorted(q["answer"])
            gold_texts = [opts[k] for k in ans]
            n_wrong = len(opts) - len(ans)
            ev = caps.get((q.get("video"), tuple(q.get("span_sec") or [])))
            if ev is None:
                raise RuntimeError("no evidence")
            clen = [len(t) for t in gold_texts]
            lo, hi = min(clen) * 0.75, max(clen) * 1.05
            gold_norm = {t.strip().lower() for t in gold_texts}

            prev_set = [v for k, v in sorted((v1.get("new_options") or opts).items())
                        if k not in (v1.get("new_answer") or ans)]
            prev_guess = ", ".join(gold_texts)
            best = None
            for rnd in range(1, a.rounds + 1):
                fb = FEEDBACK_TMPL.format(guess=prev_guess,
                                          prev="\n".join(f"  - {t}" for t in prev_set))
                new_wrong = json_call(api, REPAIR_V2.format(
                    evidence=json.dumps(ev, ensure_ascii=False), question=q["question"],
                    correct="\n".join(gold_texts), n=n_wrong, lo=int(lo), hi=int(hi),
                    feedback=fb),
                    lambda d: validate_repair(d, n_wrong, lo, hi, gold_norm))
                verdicts, ncon = verify_contradicted(api, ev, new_wrong)
                if ncon < a.require_contradicted * n_wrong and rnd < a.rounds:
                    prev_set = new_wrong
                    continue                      # plausible but unverifiable: try again
                texts = gold_texts + new_wrong
                new_opts, new_ans = shuffle_letters(texts, set(range(len(gold_texts))), q["question"])
                hits, guesses = blind(api, q["question"], new_opts, new_ans, a.votes)
                rec["rounds_used"] = rnd
                best = {"options": new_opts, "answer": new_ans,
                        "guessable": hits >= need, "contradicted": ncon,
                        "verdicts": verdicts}
                if not best["guessable"]:
                    break
                prev_set = new_wrong
                g = guesses[0] if guesses else None
                prev_guess = ", ".join(new_opts.get(k, "?") for k in (g or [])) or ", ".join(gold_texts)
            rec.update({"v2_guessable": best["guessable"], "contradicted": best["contradicted"],
                        "n_wrong": n_wrong, "options": best["options"], "answer": best["answer"],
                        "error": None})
        except Exception as exc:
            rec.update({"error": f"{type(exc).__name__}: {exc}"})
        with lock:
            done[q["id"]] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            counter["n"] += 1
            i = counter["n"]
            if i % 25 == 0:
                ok = [r for r in done.values() if not r.get("error")]
                res = sum(1 for r in ok if not r["v2_guessable"])
                el = (time.time() - t0) / 60
                print(f"  [{i}/{len(todo)}] rescued {res}/{len(ok)} = "
                      f"{100*res/max(1,len(ok)):.1f}%  {el:.0f}m", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(work, todo))
    fh.close()

    ok = [r for r in done.values() if not r.get("error")]
    n = len(ok)
    res = sum(1 for r in ok if not r["v2_guessable"])
    rlo, rhi = wilson(res, n)
    reb = sum(r["n_wrong"] for r in ok)
    con = sum(r["contradicted"] for r in ok)
    summary = {
        "targets_v1_failed": n, "errors": len(done) - n,
        "rescued_by_v2": res, "rescue_pct": round(100 * res / n, 1) if n else None,
        "rescue_ci95": [round(100 * rlo, 1), round(100 * rhi, 1)],
        "rounds_used": dict(collections.Counter(r["rounds_used"] for r in ok)),
        "contradicted_pct": round(100 * con / reb, 1) if reb else None,
        "note": ("v1 left single-choice at 73.4% blind-guessable; every row here is one "
                 "v1 failed. rescue_pct is the share v2 turns into non-guessable."),
    }
    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
