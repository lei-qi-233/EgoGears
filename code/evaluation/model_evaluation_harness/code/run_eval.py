#!/usr/bin/env python3
"""Run one model over RunningBench. Same frames, same prompt, same scoring, every model.

Two backends behind one interface:
  floodgate  -- Gemini through Apple's Floodgate gateway (mTLS + project token)
  openai     -- any OpenAI-compatible server, which is how the open-weight models are
                served with vLLM; only --base-url and --model change between them.

Option letters are re-shuffled per question (seeded by question id, so a re-run is
identical) and mapped back afterwards. Without that, a model that likes "B" scores
above chance for a reason that has nothing to do with the video.
"""
import argparse, base64, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Overridable so the exact same script runs unmodified on a remote A100 node, where the
# bundle lands at a different absolute path than it does here.
EVAL = os.environ.get("RB_EVAL_DIR", "/mnt/data/cvhci_video_understanding/eval")
QA = os.environ.get("RB_QA_DIR", "/mnt/data/cvhci_video_understanding/qa_fix")
sys.path.insert(0, QA)

PROMPT = """Answer this multiple-choice question about egocentric walking/running footage.
The frames below are sampled in order from the clip(s) the question refers to; each clip is
introduced by its label.

Watch before deciding. Do not answer from the wording of the options alone.

QUESTION: {question}

OPTIONS:
{options}

Select exactly {n} option{plural}.
Return ONLY JSON: {{"answer": [{example}]}}"""


def shuffled_view(options, seed):
    """Return (displayed -> text, displayed -> real letter)."""
    real = sorted(options)
    order = list(real)
    random.Random(seed).shuffle(order)
    disp = {}
    back = {}
    for i, r in enumerate(order):
        d = real[i]
        disp[d] = options[r]
        back[d] = r
    return disp, back


def parse_answer(raw, letters):
    import re
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
    ans = (d or {}).get("answer") if isinstance(d, dict) else None
    if ans is None:
        ans = re.findall(r'"([A-J])"', raw or "") or re.findall(r"\b([A-J])\b", raw or "")
    if isinstance(ans, str):
        ans = [ans]
    if not isinstance(ans, list):
        # 有些模型偶尔吐出 {"answer": 5} 这种非法格式(数字而不是字母列表)，
        # 不兜底的话 `for a in ans` 直接 TypeError，整个 run_eval.py 崩溃退出
        # (2026-09-17 实测 ERNIE-4.5-VL-28B-A3B 跑到 401/698 就这样整体挂掉)。
        ans = []
    return sorted({a.strip()[0].upper() for a in ans if isinstance(a, str) and a.strip()} & set(letters))


class Floodgate:
    def __init__(self, model, rps):
        from floodgate import Floodgate as FG, RateLimiter
        self.api = FG(limiter=RateLimiter(rps))
        self.model = model

    def ask(self, text, images):
        parts = [{"text": text}]
        for lab, files in images:
            parts.append({"text": f"--- {lab} ---"})
            for f in files:
                parts.append({"inlineData": {"mimeType": "image/jpeg",
                                             "data": base64.b64encode(open(f, "rb").read()).decode("ascii")}})
        return self.api.generate(self.model, parts, max_tokens=2048, timeout=600,
                                 attempts=4, temperature=0.0)


class OpenAICompat:
    def __init__(self, model, base_url, rps, api_key="EMPTY", enable_thinking=None, max_tokens=2048):
        import requests
        from floodgate import RateLimiter
        self.s = requests.Session(); self.s.trust_env = False
        self.limiter = RateLimiter(rps)
        self.model, self.url, self.key = model, base_url.rstrip("/") + "/chat/completions", api_key
        # A handful of models (Qwen3.5-*, ERNIE-4.5-VL, GLM-4.6V, Gemma-4-*) carry a single
        # checkpoint with a chat-template-level thinking toggle rather than a separate
        # -Thinking release. Left unset, several of them DEFAULT TO THINKING ON and mix the
        # reasoning trace into the same `content` field vLLM returns -- confirmed live
        # against Qwen3.5-9B on 2026-09-17: an unrelated 3-option question came back as an
        # 822-char "Thinking Process:" essay before ever reaching the JSON answer. At this
        # class's 2048-token cap that trace can consume the whole budget on a real 64-frame
        # question, truncating the JSON answer before it starts -- which is what the
        # "wrong number of options selected" pattern earlier turned out to be, not the
        # model actually miscounting. `enable_thinking=None` leaves the model's own default
        # untouched (for models with no such toggle); explicit True/False sets
        # `chat_template_kwargs` the same way vLLM's OpenAI server documents it.
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens

    def ask(self, text, images):
        content = [{"type": "text", "text": text}]
        for lab, files in images:
            content.append({"type": "text", "text": f"--- {lab} ---"})
            for f in files:
                b = base64.b64encode(open(f, "rb").read()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": self.max_tokens, "temperature": 0.0}
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        last = None
        for k in range(4):
            self.limiter.acquire()
            try:
                r = self.s.post(self.url, json=body, timeout=900,
                                headers={"Authorization": f"Bearer {self.key}"})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
                time.sleep(min(60, 4 * 2 ** k))
        raise RuntimeError(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["floodgate", "openai"], default="floodgate")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tag", help="output name; defaults to the model name")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rps", type=float, default=0.45)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="run only questions where index %% nshards == shard")
    ap.add_argument("--nshards", type=int, default=1, help="split the corpus across this many parallel replicas")
    ap.add_argument("--enable-thinking", choices=["true", "false"], default=None,
                    help="for models with a chat-template thinking toggle (Qwen3.5-*, ERNIE-4.5-VL, "
                         "GLM-4.6V, Gemma-4-*): force it on/off via chat_template_kwargs. Omit to "
                         "leave the model's own default untouched.")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="raise this when --enable-thinking=true: the reasoning trace shares this "
                         "budget with the JSON answer and will truncate it if too small")
    a = ap.parse_args()
    tag = a.tag or a.model.replace("/", "_")
    if a.nshards > 1:
        tag = f"{tag}.shard{a.shard}of{a.nshards}"
    out_path = f"{EVAL}/results/{tag}.jsonl"
    os.makedirs(f"{EVAL}/results", exist_ok=True)

    corpus = [json.loads(l) for l in open(f"{QA}/runningbench_v2_kept.jsonl")]
    frames = {}
    for l in open(f"{EVAL}/frames_index.jsonl"):
        r = json.loads(l)
        if r.get("error"):
            continue
        # frames_index.jsonl bakes in the absolute path from wherever extract_frames.py
        # was run; on a remote node the bundle lands under a different root, so rebuild
        # each path from EVAL rather than trust the recorded one. Layout is fixed:
        # <EVAL>/frames/<review_id>/<basename>.
        rid = r["review_id"]
        for c in r["clips"]:
            c["frames"] = [f"{EVAL}/frames/{rid}/{os.path.basename(fp)}" for fp in c["frames"]]
        frames[rid] = r
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                x = json.loads(l)
                if not x.get("error"):
                    done.add(x["review_id"])
            except Exception:
                pass
    todo = [q for q in corpus if q["review_id"] in frames and q["review_id"] not in done]
    if a.nshards > 1:
        # stable order (corpus file order) then take every nshards-th question, so two
        # replicas covering different shards never duplicate or skip work
        todo = todo[a.shard::a.nshards]
    if a.limit:
        todo = todo[: a.limit]
    print(f"model={a.model} backend={a.backend} corpus={len(corpus)} todo={len(todo)}", flush=True)

    think = {"true": True, "false": False, None: None}[a.enable_thinking]
    client = (Floodgate(a.model, a.rps) if a.backend == "floodgate"
              else OpenAICompat(a.model, a.base_url, a.rps, enable_thinking=think, max_tokens=a.max_tokens))

    def one(q):
        rid = q["review_id"]
        fr = frames[rid]
        disp, back = shuffled_view(q["options"], rid)
        letters = sorted(disp)
        n = q["n_select"]
        text = PROMPT.format(question=q["question"],
                             options="\n".join(f"{l}. {disp[l]}" for l in letters),
                             n=n, plural="s" if n > 1 else "",
                             example=", ".join(f'"{l}"' for l in letters[:n]))
        images = [(c["label"].split()[0], c["frames"]) for c in fr["clips"]]
        t0 = time.time()
        try:
            raw = client.ask(text, images)
        except Exception as exc:
            return {"review_id": rid, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        try:
            picked = parse_answer(raw, letters)
            mapped = sorted({back[l] for l in picked if l in back})
            gold = sorted(q["answer"])
        except Exception as exc:
            # 解析阶段本身出错(比如模型偶尔吐出畸形 JSON)不该让整条流水线崩掉——
            # 之前这里没兜底，ERNIE-4.5-VL-28B-A3B 跑到 401/698 撞见一次就整体退出了。
            return {"review_id": rid, "raw": raw, "error": f"parse:{type(exc).__name__}: {str(exc)[:200]}"}
        return {"review_id": rid, "unit": q["unit"], "question_type": q["question_type"],
                "n_select": n, "n_options": len(q["options"]), "n_frames": fr["n_frames"],
                "n_clips": len(fr["clips"]), "model": a.model, "raw": raw,
                "pred": mapped, "gold": gold, "exact": mapped == gold,
                "overlap": len(set(mapped) & set(gold)) / max(1, len(gold)),
                "n_pred": len(mapped), "latency_s": round(time.time() - t0, 1)}

    lock = threading.Lock()
    n = ok = exact = 0
    with open(out_path, "a") as fh, ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(one, q) for q in todo]):
            r = f.result()
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
                n += 1
                if not r.get("error"):
                    ok += 1; exact += r["exact"]
                if n % 50 == 0:
                    print(f"{n}/{len(todo)} ok={ok} exact={exact} ({100*exact/max(1,ok):.1f}%)", flush=True)
    print(f"DONE {n} ok={ok} exact={exact} ({100*exact/max(1,ok):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
