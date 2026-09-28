"""Minimal Floodgate (Gemini) client — rebuilt per FLOODGATE_ANNOTATION.md §2.

This is the module `build_fullvideo_annotations.py` used to load dynamically from
`repair_runningbench_annotations.py`, which was never backed up to Conductor.
"""
import json, os, random, re, threading, time
import requests

FLOODGATE_ROOT = "https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models"
CERT = ("/turibolt_k8s_mounts/narrative/turi/cert.pem",
        "/turibolt_k8s_mounts/narrative/turi/private.pem")

CAPTION_MODEL = "gemini-3.1-pro-preview"   # every call that looks at video
STRUCTURE_MODEL = "gemini-3.5-flash"       # blind gate / distractor rewrite
QUOTA_PER_SECOND = 0.45


class RateLimiter:
    """One global token bucket; a per-thread limiter multiplies the quota by the
    number of threads."""
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
    def __init__(self, token=None, limiter=None):
        self.token = token if token is not None else os.environ.get("FLOODGATE_PROJECT_TOKEN", "")
        self.limiter = limiter or RateLimiter(QUOTA_PER_SECOND)
        self.session = requests.Session()
        self.session.trust_env = False        # proxies break the mTLS handshake
        self.lock = threading.Lock()

    def generate(self, model, parts, max_tokens=16384, attempts=5, timeout=600,
                 temperature=0.2, schema=None, throttle_retries=8):
        url = f"{FLOODGATE_ROOT}/{model}:generateContent"
        gen = {"maxOutputTokens": max_tokens, "temperature": temperature,
               "responseMimeType": "application/json"}
        if schema:
            gen["responseSchema"] = schema
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
        headers = {"X-Floodgate-Project-Token": self.token, "Content-Type": "application/json"}
        last = "unknown"
        k = throttled = 0
        while k < attempts:
            self.limiter.acquire()
            status = retry_after = None
            try:
                with self.lock:
                    sess = self.session
                r = sess.post(url, headers=headers, json=payload, cert=CERT, timeout=timeout)
                status = r.status_code
                if status in (429, 503):
                    try:
                        retry_after = float(r.headers.get("Retry-After") or 0) or None
                    except ValueError:
                        retry_after = None
                r.raise_for_status()
                body = r.json()
                cands = body.get("candidates") or [{}]
                texts = [p["text"] for p in cands[0].get("content", {}).get("parts", [])
                         if "text" in p]
                if texts:
                    return "\n".join(texts)
                last = f"empty candidates: {str(body)[:300]}"
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:300]}"
            # A burst of 429s poisons this session's pool permanently: every later
            # call then fails for reasons unrelated to the request. Rebuild it
            # instead of spending the remaining retries on a dead socket.
            if throttled + k >= 2:
                with self.lock:
                    try:
                        self.session.close()
                    except Exception:
                        pass
                    self.session = requests.Session()
                    self.session.trust_env = False
            # 429/503 is the server saying "later", not "no". Spending one of the five
            # semantic attempts on it is what lost 41 repair attempts on 2026-09-16 --
            # every one of them a question that was never actually judged. Wait it out
            # on a separate budget instead.
            if status in (429, 503) and throttled < throttle_retries:
                throttled += 1
                time.sleep((retry_after or min(120.0, 8.0 * 2 ** min(throttled, 4)))
                           + random.random() * 3)
                continue
            k += 1
            if k < attempts:
                time.sleep(min(60, 4 * (2 ** min(k, 4))) + random.random() * 2)
        raise RuntimeError(last)


def clean_json(raw):
    """Three real failure modes: ```json fences, array wrapping, truncation."""
    d = None
    try:
        d = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            try:
                d = json.loads(m.group(0))
            except Exception:
                d = None
    if isinstance(d, list):
        d = next((x for x in d if isinstance(x, dict)), None)
    if isinstance(d, dict):
        return d
    m = re.search(r'"verdict"\s*:\s*"(clear|ambiguous|cant_tell|supported|contradicted|insufficient)"',
                  raw or "")
    if m:
        return {"verdict": m.group(1), "truncated": True}
    return None


def generate_valid_json(api, model, parts, validate, attempts=3, **kw):
    """Call until clean_json gives a dict that passes `validate`."""
    last = None
    for _ in range(attempts):
        try:
            raw = api.generate(model, parts, **kw)
        except Exception as exc:
            last = f"generate failed: {exc}"
            continue
        d = clean_json(raw)
        if d is None:
            last = f"unparseable: {str(raw)[:200]}"
            continue
        err = validate(d)
        if err is None:
            return d
        last = f"invalid: {err}"
    raise RuntimeError(last or "no attempts")


def load_progress(path):
    """Append-only JSONL resume: a killed process leaves a half line, skip it."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("error") and r.get("id"):
                out[r["id"]] = r
    return out
