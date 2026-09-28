#!/usr/bin/env python3
"""Blind-guess test for the Video Bench corpus (1,892 MCQ).

Runs the SAME procedure the full-video gates use -- no video, question+options
only, three votes, guessable at >=2/3 -- over a stratified sample, and reports
Wilson intervals overall and split by family and option count.

Why this exists: Video Bench has never faced a blind model. Its published
"blind_ceiling 33.2" is an always-answer-the-same-letter ceiling, not a measured
blind rate. On the segment corpus that same confusion understated the truth by
nearly 3x (ceiling 17.6% vs measured 48.6%).

Option count matters: 444 of these items have only 2 options (random = 50%), so
a headline number alone is misleading. Always read the by-n_options table.
"""
import argparse, collections, json, math, os, random, re, sys, threading, time
from pathlib import Path
import requests

FLOODGATE_ROOT = "https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models"
CERT = ("/turibolt_k8s_mounts/narrative/turi/cert.pem",
        "/turibolt_k8s_mounts/narrative/turi/private.pem")
MODEL = "gemini-3.5-flash"
QUOTA_PER_SECOND = 0.45
SEED = 20260830

# Identical to BLIND_PROMPT in build_fullvideo_annotations.py.
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
            delay = min(60, 4 * (2 ** min(k, 4))) + random.random() * 2
            print(f"    retry {k+1}/{attempts} in {delay:.0f}s: {last[:110]}", flush=True)
            time.sleep(delay)
        raise RuntimeError(last)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).strip().lower()).strip()


def parse_answer(raw, options):
    """Pull the chosen option out of the model's JSON, tolerating letters and prose."""
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
    if isinstance(a, list):
        a = a[0] if a else None
    if a is None:
        return None
    na = norm(a)
    opts_n = [norm(o) for o in options]
    if na in opts_n:
        return options[opts_n.index(na)]
    # bare letter -> option by position
    m = re.fullmatch(r"([a-z])[).:]?", na)
    if m:
        i = ord(m.group(1)) - ord("a")
        if 0 <= i < len(options):
            return options[i]
    for i, o in enumerate(opts_n):          # substring fallback
        if o and (o in na or na in o):
            return options[i]
    return None


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def stratified(items, n, seed=SEED):
    """Proportional allocation over (family, n_options), remainder by largest fraction."""
    rnd = random.Random(seed)
    buckets = collections.defaultdict(list)
    for it in items:
        buckets[(it.get("family"), it.get("n_options"))].append(it)
    total = len(items)
    quota, frac = {}, {}
    for k, v in buckets.items():
        exact = len(v) * n / total
        quota[k] = int(exact)
        frac[k] = exact - int(exact)
    short = n - sum(quota.values())
    for k in sorted(frac, key=lambda x: -frac[x])[:short]:
        quota[k] += 1
    out = []
    for k, v in buckets.items():
        take = min(quota.get(k, 0), len(v))
        out.extend(rnd.sample(v, take))
    rnd.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--out", default="/mnt/task_runtime/blind_test/videobench_blind.json")
    ap.add_argument("--checkpoint", default="/mnt/task_runtime/blind_test/videobench_blind.progress.jsonl")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.questions) if l.strip()]
    items = [x for x in items if x.get("options") and x.get("answer") is not None]
    sample = stratified(items, min(a.n, len(items)))
    print(f"corpus={len(items)}  sample={len(sample)}  votes={a.votes}  model={MODEL}", flush=True)

    done = {}
    cp = Path(a.checkpoint)
    if cp.exists():
        for line in cp.open():
            try:
                r = json.loads(line)
                done[r["id"]] = r
            except Exception:
                pass
        print(f"resuming: {len(done)} already done", flush=True)

    api = Floodgate(os.environ.get("FLOODGATE_PROJECT_TOKEN", ""), RateLimiter(QUOTA_PER_SECOND))
    fh = cp.open("a")
    t0 = time.time()
    for i, q in enumerate(sample, 1):
        if q["id"] in done:
            continue
        opts = q["options"]
        arity = "Exactly one option is correct."
        guesses, hits = [], 0
        err = None
        for t in range(a.votes):
            try:
                raw = api.generate(BLIND_PROMPT.format(
                    question=q["question"],
                    options=json.dumps(opts, ensure_ascii=False),
                    arity=arity + f" (attempt {t+1})"))
                g = parse_answer(raw, opts)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                g = None
            guesses.append(g)
            if g is not None and norm(g) == norm(q["answer"]):
                hits += 1
        rec = {"id": q["id"], "family": q.get("family"), "n_options": q.get("n_options"),
               "answer": q["answer"], "guesses": guesses, "hits": hits,
               "guessable": hits >= 2, "all_correct": hits == a.votes, "error": err}
        done[q["id"]] = rec
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        if i % 25 == 0 or i == len(sample):
            el = time.time() - t0
            n_done = len([r for r in done.values()])
            gz = sum(1 for r in done.values() if r["guessable"])
            print(f"  [{i}/{len(sample)}] guessable so far {gz}/{n_done} = "
                  f"{100*gz/max(1,n_done):.1f}%  elapsed {el/60:.1f}m", flush=True)
    fh.close()

    res = [done[q["id"]] for q in sample if q["id"] in done]
    ok = [r for r in res if r["error"] is None or r["hits"] > 0]
    n = len(res)
    gz = sum(1 for r in res if r["guessable"])
    allc = sum(1 for r in res if r["all_correct"])
    lo, hi = wilson(gz, n)

    def table(key):
        rows = []
        by = collections.defaultdict(list)
        for r in res:
            by[r[key]].append(r)
        for k, v in sorted(by.items(), key=lambda x: -len(x[1])):
            g = sum(1 for r in v if r["guessable"])
            l, h = wilson(g, len(v))
            rows.append({str(key): k, "n": len(v), "guessable": g,
                         "rate": round(100 * g / len(v), 1),
                         "ci95": [round(100 * l, 1), round(100 * h, 1)]})
        return rows

    # Random baseline given the sample's option-count mix.
    base = sum(1.0 / r["n_options"] for r in res if r.get("n_options")) / max(1, n)

    out = {"model": MODEL, "votes": a.votes, "corpus_size": len(items), "sample_size": n,
           "guessable_ge2of3": gz, "guessable_rate_pct": round(100 * gz / n, 1),
           "ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
           "all_votes_correct": allc, "all_votes_correct_pct": round(100 * allc / n, 1),
           "random_baseline_pct": round(100 * base, 1),
           "extrapolated_guessable_in_corpus": round(len(items) * gz / n),
           "by_family": table("family"), "by_n_options": table("n_options"),
           "errors": sum(1 for r in res if r["error"])}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
