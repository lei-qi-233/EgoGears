#!/usr/bin/env python3
"""Blind-guess gate for the segments_60s corpus (8,033 questions, 100% ungated).

Runs gate ④ from the full-video pipeline -- no video, question+options only,
three votes, rejected at >=2/3 -- over the whole segment corpus, and writes a
gated corpus so the "never screened" backlog actually shrinks.

Only the blind gate runs here: the visual re-check gate needs the 117 GiB of
source video, which is on Conductor and not mounted locally. The blind gate is
the one that does the most work anyway -- it produced 542 of the 1,085 rejections
recorded in the release manifest.

The distractor-repair overlay is applied first, so each question is judged in the
form it would actually ship in. Rows whose repair was rolled back
(fully_reverted) keep their original options.
"""
import argparse, collections, json, math, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

FLOODGATE_ROOT = "https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models"
CERT = ("/turibolt_k8s_mounts/narrative/turi/cert.pem",
        "/turibolt_k8s_mounts/narrative/turi/private.pem")
MODEL = "gemini-3.5-flash"
QUOTA_PER_SECOND = 0.45

# Verbatim from build_fullvideo_annotations.py.
BLIND_PROMPT = """Answer this multiple-choice question WITHOUT any video — you have none. Return only
JSON: {{"answer": ["..."]}}

QUESTION: {question}
OPTIONS: {options}
{arity}"""


class RateLimiter:
    def __init__(self, per_second):
        self.interval, self.lock = 1.0 / per_second, threading.Lock()
        self.next_slot = time.monotonic()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_slot - now)
            self.next_slot = max(now, self.next_slot) + self.interval
        if wait:
            time.sleep(wait)


class Floodgate:
    def __init__(self, token, limiter):
        self.token, self.limiter = token, limiter
        self.session = requests.Session()
        self.session.trust_env = False

    def generate(self, prompt, max_tokens=4096, attempts=6):
        url = f"{FLOODGATE_ROOT}/{MODEL}:generateContent"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2,
                                        "responseMimeType": "application/json"}}
        headers = {"X-Floodgate-Project-Token": self.token, "Content-Type": "application/json"}
        last = "unknown"
        for k in range(attempts):
            self.limiter.acquire()
            try:
                r = self.session.post(url, headers=headers, json=payload, cert=CERT, timeout=180)
                r.raise_for_status()
                body = r.json()
                texts = [p["text"] for p in body.get("candidates", [{}])[0]
                         .get("content", {}).get("parts", []) if "text" in p]
                if texts:
                    return "\n".join(texts)
                last = f"empty candidates: {str(body)[:200]}"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            # A burst of 429s can leave this session's pooled connections unusable for the
            # rest of a multi-hour run, so every later call fails for a reason that has
            # nothing to do with the request. Rebuild the session rather than spending the
            # remaining attempts on a socket that will not recover.
            if k >= 2:
                try:
                    self.session.close()
                except Exception:
                    pass
                self.session = requests.Session()
                self.session.trust_env = False
            time.sleep(min(60, 4 * (2 ** min(k, 4))) + random.random() * 2)
        raise RuntimeError(last)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).strip().lower()).strip()


def to_letters(ans, options):
    """Normalise a model answer to a sorted letter list. Accepts letters or option text."""
    if ans is None:
        return None
    if not isinstance(ans, list):
        ans = [ans]
    text2letter = {norm(v): k for k, v in options.items()}
    out = []
    for a in ans:
        s = str(a).strip()
        m = re.match(r"^\s*([A-Za-z])\s*[).:]?\s*$", s)
        if m and m.group(1).upper() in options:
            out.append(m.group(1).upper())
            continue
        n = norm(s)
        if n in text2letter:
            out.append(text2letter[n])
            continue
        hit = [k for t, k in text2letter.items() if t and (t in n or n in t)]
        if len(hit) == 1:
            out.append(hit[0])
    return sorted(set(out)) if out else None


def parse(raw, options):
    try:
        d = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except Exception:
            return None
    a = d.get("answer") if isinstance(d, dict) else d
    return to_letters(a, options)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load_corpus(qdir):
    base = [json.loads(l) for l in open(f"{qdir}/segments_60s.jsonl") if l.strip()]
    overlay = {}
    for l in open(f"{qdir}/segments_60s_repaired_distractors.jsonl"):
        if not l.strip():
            continue
        r = json.loads(l)
        if not r.get("fully_reverted"):          # rolled-back repairs keep the original
            overlay[r["id"]] = r
    n_patched = 0
    for q in base:
        r = overlay.get(q.get("id"))
        if r:
            q["options"], q["answer"] = r["options"], r["answer"]
            q["repaired"] = True
            n_patched += 1
        else:
            q["repaired"] = False
    return base, n_patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent questions; the shared limiter still caps total call rate")
    ap.add_argument("--out", default="/mnt/task_runtime/blind_test/segments_60s_gated.jsonl")
    ap.add_argument("--summary", default="/mnt/task_runtime/blind_test/segments_60s_gate_summary.json")
    ap.add_argument("--checkpoint", default="/mnt/task_runtime/blind_test/segments_60s_gate.progress.jsonl")
    a = ap.parse_args()

    corpus, n_patched = load_corpus(a.qdir)
    if a.limit:
        corpus = corpus[:a.limit]
    print(f"corpus={len(corpus)}  repair-overlay applied to {n_patched}  votes={a.votes}", flush=True)

    done = {}
    cp = Path(a.checkpoint)
    if cp.exists():
        for line in cp.open():
            try:
                r = json.loads(line)
                done[r["id"]] = r
            except Exception:
                pass
        print(f"resuming: {len(done)} already gated", flush=True)

    # One shared limiter caps the whole pool at QUOTA_PER_SECOND. Workers only hide
    # per-call latency: sequentially we reached 0.185 calls/s against a 0.45 ceiling,
    # so the run was latency-bound, not quota-bound.
    api = Floodgate(os.environ.get("FLOODGATE_PROJECT_TOKEN", ""), RateLimiter(QUOTA_PER_SECOND))
    fh = cp.open("a")
    lock = threading.Lock()
    t0 = time.time()
    todo = [q for q in corpus if q.get("id") not in done]
    counter = {"n": 0}

    def gate_one(q):
        opts = q["options"]
        gold = sorted(q["answer"])
        arity = ("Exactly one option is correct." if len(gold) == 1
                 else "More than one option is correct.")
        guesses, hits, err = [], 0, None
        for t in range(a.votes):
            try:
                g = parse(api.generate(BLIND_PROMPT.format(
                    question=q["question"], options=json.dumps(opts, ensure_ascii=False),
                    arity=arity + f" (attempt {t+1})")), opts)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                g = None
            guesses.append(g)
            hits += (g == gold)
        rec = {"id": q["id"], "arity": q.get("arity"), "n_options": q.get("n_options"),
               "repaired": q.get("repaired"), "gold": gold, "guesses": guesses,
               "hits": hits, "blind_guessable": hits >= 2, "error": err}
        with lock:
            done[q["id"]] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            counter["n"] += 1
            i = counter["n"]
            if i % 100 == 0:
                n = len(done)
                bad = sum(1 for r in done.values() if r["blind_guessable"])
                el = (time.time() - t0) / 60
                rate = i / el if el else 0
                eta = (len(todo) - i) / rate / 60 if rate else 0
                print(f"  [{i}/{len(todo)}] blind-guessable {bad}/{n} = {100*bad/max(1,n):.1f}%  "
                      f"{el:.0f}m elapsed  {rate:.1f}/min  ETA {eta:.1f}h", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(gate_one, todo))
    fh.close()

    # ---- write the gated corpus ----
    kept = rejected = 0
    with open(a.out, "w") as out:
        for q in corpus:
            r = done.get(q.get("id"))
            if not r:
                continue
            q["gates"] = {"blind_guess": {"guesses": r["guesses"], "hits_of_3": r["hits"],
                                          "matches_gold": r["blind_guessable"]}}
            if r["blind_guessable"]:
                q["screening_status"] = "rejected_blind_guessable"
                rejected += 1
            else:
                q["screening_status"] = "passed_blind_gate"
                kept += 1
            out.write(json.dumps(q, ensure_ascii=False) + "\n")

    res = list(done.values())
    n = len(res)
    bad = sum(1 for r in res if r["blind_guessable"])
    lo, hi = wilson(bad, n)

    def table(key):
        by = collections.defaultdict(list)
        for r in res:
            by[r[key]].append(r)
        rows = []
        for k, v in sorted(by.items(), key=lambda x: -len(x[1])):
            b = sum(1 for r in v if r["blind_guessable"])
            l, h = wilson(b, len(v))
            rows.append({str(key): k, "n": len(v), "blind_guessable": b,
                         "rate": round(100 * b / len(v), 1),
                         "ci95": [round(100 * l, 1), round(100 * h, 1)]})
        return rows

    summary = {"model": MODEL, "votes": a.votes, "gated": n,
               "blind_guessable": bad, "rate_pct": round(100 * bad / n, 1),
               "ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
               "passed_blind_gate": kept, "rejected": rejected,
               "all_votes_correct": sum(1 for r in res if r["hits"] == a.votes),
               "by_arity": table("arity"), "by_n_options": table("n_options"),
               "by_repaired": table("repaired"),
               "errors": sum(1 for r in res if r["error"])}
    Path(a.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
