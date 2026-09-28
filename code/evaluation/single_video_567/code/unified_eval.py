#!/usr/bin/env python3
"""Evaluate both RunningBench QA sets with the original 64-frame protocol.

Uses an OpenAI-compatible vision chat endpoint (for example, a local vLLM server).
No model is downloaded or started by this script.
"""

import argparse
import base64
import collections
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VERSION = "runningbench-unified-v8"
# Frame extraction is unchanged since v7; keep its cache key so extracted frames are reused.
FRAME_VERSION = "runningbench-unified-v7"
DATASETS = {
    "single_698": "segments_60s_gemini_verification/final/segments_700_verified.jsonl",
    "single_567_human": "human_answer_repair_567/single_567_human_eval.jsonl",
    "mixed_1487": "qa_fix_gold_recovery_and_adjudication/data/runningbench_v2_kept.jsonl",
}
MULTI_VIDEO_PROMPT = """Answer this multiple-choice question about egocentric walking/running footage.
The frames below are sampled in order from the clip(s) the question refers to; each clip is
introduced by its label.

Watch before deciding. Do not answer from the wording of the options alone.

QUESTION: {question}

OPTIONS:
{options}

Select exactly {n} option{plural}.
Return ONLY JSON: {{"answer": [{example}]}}"""
SINGLE_VIDEO_PROMPT = """Answer this multiple-choice question about egocentric walking/running footage.
The frames below are sampled in chronological order from the single video clip (at most 60 seconds long) associated with this question.

Watch the video frames before deciding. Base your answer on visual evidence from this video. Do not answer from the wording of the options alone.

QUESTION: {question}

OPTIONS:
{options}

Select exactly {n} option{plural}.
Return ONLY JSON: {{"answer": [{example}]}}"""
MULTI_UNITS = {"cross_video", "cross_recording", "anonymous_multi_clip"}


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def question_id(q, dataset):
    return q["id"] if dataset.startswith("single_") else q["review_id"]


def load_questions(root, dataset):
    path = root / DATASETS[dataset]
    questions = list(jsonl(path))
    ids = [question_id(q, dataset) for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {path}")
    return path, questions


def shuffled_view(options, seed):
    real = sorted(options)
    order = list(real)
    random.Random(seed).shuffle(order)
    shown = {real[i]: options[order[i]] for i in range(len(real))}
    back = {real[i]: order[i] for i in range(len(real))}
    return shown, back


ANSWER_ITEM = re.compile(r"^(?:option\s+)?\(?([A-Z])\)?[.):]?$", re.I)


def parse_answer(raw, letters):
    """Return (displayed letters, error) from the last JSON object that has an "answer" key.

    Strict by design: no free-text letter fallback, and a letter outside the shown options
    is an error rather than being silently dropped.
    """
    text = raw or ""
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            found = obj
    if found is None:
        return None, 'No JSON object with an "answer" key'
    answer = found["answer"]
    if isinstance(answer, str):
        answer = re.split(r"\s*[,;/]\s*", answer.strip())
    if not isinstance(answer, list):
        return None, f"Answer is not a list: {answer!r}"[:200]
    chosen = set()
    for item in answer:
        match = ANSWER_ITEM.match(item.strip()) if isinstance(item, str) else None
        if not match:
            return None, f"Unrecognized answer item: {item!r}"[:200]
        letter = match.group(1).upper()
        if letter not in letters:
            return None, f"Answer letter {letter} is not among the shown options"
        chosen.add(letter)
    if not chosen:
        return None, "Empty answer"
    return sorted(chosen), None


def split_reasoning(raw):
    """Separate a <think> trace from the final answer text.

    Handles templates that pre-fill the opening <think> so only </think> is generated.
    """
    if "</think>" not in raw:
        return None, raw
    head, answer_text = raw.rsplit("</think>", 1)
    if "<think>" in head:
        head = head.split("<think>", 1)[1]
    return head.strip() or None, answer_text


def allocate(durations, total, floor):
    """Match model_evaluation_harness/code/extract_frames.py."""
    n = len(durations)
    if not n:
        raise ValueError("Question has no clips")
    if n * floor >= total:
        return [floor] * n
    summed = sum(durations) or 1.0
    raw = [d / summed * total for d in durations]
    base = [max(floor, int(x)) for x in raw]
    while sum(base) > total:
        i = max(range(n), key=lambda j: (base[j] - floor, raw[j]))
        if base[i] <= floor:
            break
        base[i] -= 1
    remainder = total - sum(base)
    order = sorted(range(n), key=lambda j: -(raw[j] - int(raw[j])))
    for i in range(remainder):
        base[order[i % n]] += 1
    return base


def video_duration(path):
    result = subprocess.run(
        ["ffmpeg", "-i", str(path)], capture_output=True, timeout=120,
    )
    match = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)",
                      result.stderr.decode("utf-8", "replace"))
    if not match:
        return 60.0
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


class MediaResolver:
    def __init__(self, root, single_clips, cache):
        self.root, self.single_clips, self.cache = root, single_clips, cache
        self.tars = {}

    def close(self):
        for archive in self.tars.values():
            archive.close()

    def clips(self, dataset, question):
        if dataset.startswith("single_"):
            path = self.single_clips / f"{question['id']}.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
            return [("CLIP_1", path)]

        bundle = question["bundle"]
        archive = self.tars.get(bundle)
        if archive is None:
            path = self.root / "annotation_bundles" / f"{bundle}.tar"
            archive = tarfile.open(path, "r")
            self.tars[bundle] = archive
        resolved = []
        for clip in question["clips"]:
            member_name = f"{bundle}/{clip['path']}"
            member = archive.getmember(member_name)
            if not member.isfile():
                raise ValueError(f"Not a media file: {member_name}")
            target = self.cache / bundle / question["review_id"] / Path(clip["path"]).name
            if not target.is_file() or target.stat().st_size != member.size:
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(target.suffix + ".part")
                with archive.extractfile(member) as source, open(temp, "wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
                os.replace(temp, target)
            resolved.append((clip["label"], target))
        return resolved


def extract_question_frames(dataset, qid, clips, cache, budget=64, floor=4, width=480):
    durations = [video_duration(path) for _, path in clips]
    quotas = allocate(durations, budget, floor)
    base = cache / dataset / f"f{budget}_m{floor}_w{width}" / qid
    all_clips = []
    for i, ((label, source), duration, quota) in enumerate(zip(clips, durations, quotas)):
        directory = base / f"c{i:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "frames.json"
        expected = {"version": FRAME_VERSION, "source": str(source), "source_size": source.stat().st_size,
                    "quota": quota, "width": width}
        frames = []
        if marker.is_file():
            try:
                previous = json.loads(marker.read_text(encoding="utf-8"))
                if previous["config"] == expected:
                    frames = [Path(p) for p in previous["frames"]]
                    if not frames or not all(p.is_file() for p in frames):
                        frames = []
            except (KeyError, ValueError):
                pass
        if not frames:
            for stale in directory.glob("frame_*.jpg"):
                stale.unlink()
            fps = quota / max(duration, 0.1)
            # Intentionally identical to the original mixed-set harness: width <= 480,
            # with aspect ratio preserved. Portrait frames can have height > 480.
            command = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1", "-i", str(source),
                       "-vf", f"fps={fps:.6f},scale='min({width},iw)':-2", "-frames:v", str(quota),
                       "-q:v", "3", str(directory / "frame_%03d.jpg")]
            subprocess.run(command, check=True, capture_output=True, timeout=900)
            frames = sorted(directory.glob("frame_*.jpg"))
            if not frames:
                raise RuntimeError(f"ffmpeg generated no frames from {source}")
            marker.write_text(json.dumps({"config": expected, "frames": [str(p) for p in frames]}), encoding="utf-8")
        all_clips.append((label, frames))
    return all_clips, quotas


class RateLimiter:
    def __init__(self, requests_per_second):
        self.interval = 1 / max(requests_per_second, 0.01)
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            delay = self.interval - (time.monotonic() - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


class Client:
    def __init__(self, base_url, model, api_key, thinking, max_tokens, max_tokens_cap, rps):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.thinking = model, api_key, thinking
        self.max_tokens, self.max_tokens_cap = max_tokens, max_tokens_cap
        self.limiter = RateLimiter(rps)

    def ask(self, prompt, clips):
        content = [{"type": "text", "text": prompt}]
        for label, frames in clips:
            content.append({"type": "text", "text": f"--- {label} ---"})
            for path in frames:
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0.0}
        if self.thinking != "auto":
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking == "true"}
        attempts = []
        budget = self.max_tokens
        while True:
            body["max_tokens"] = budget
            payload = json.dumps(body).encode("utf-8")
            last = None
            result = None
            for request_attempt in range(4):
                self.limiter.wait()
                request = urllib.request.Request(
                    self.url, data=payload, method="POST",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
                try:
                    with urllib.request.urlopen(request, timeout=900) as response:
                        result = json.load(response)
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read(1000).decode("utf-8", "replace")
                    last = f"HTTP {exc.code}: {detail}"
                    if exc.code not in (429, 500, 502, 503, 504):
                        break
                except (urllib.error.URLError, ValueError) as exc:
                    last = f"{type(exc).__name__}: {exc}"
                if request_attempt < 3:
                    time.sleep(min(30, 2 ** (request_attempt + 1)))
            if result is None:
                if attempts:
                    return {"error": f"Request failed after truncated response: {last}", "attempts": attempts}
                raise RuntimeError(f"Model request failed: {last}")
            choice = result["choices"][0]
            item = {"message": choice["message"], "finish_reason": choice.get("finish_reason"),
                    "usage": result.get("usage"), "max_tokens": budget}
            attempts.append(item)
            if item["finish_reason"] != "length":
                return {**item, "attempts": attempts,
                        "error": None if item["finish_reason"] == "stop"
                        else f"Unverified finish_reason={item['finish_reason']!r}"}
            if budget >= self.max_tokens_cap:
                return {**item, "attempts": attempts,
                        "error": f"Output truncated at max_tokens_cap={self.max_tokens_cap}"}
            budget = min(self.max_tokens_cap, budget * 2)


class FloodgateClient:
    def __init__(self, root, model, max_tokens, rps):
        sys.path.insert(0, str(root / "model_evaluation_harness" / "code"))
        from floodgate import Floodgate, RateLimiter

        self.client = Floodgate(limiter=RateLimiter(rps))
        self.model, self.max_tokens = model, max_tokens

    def ask(self, prompt, clips):
        parts = [{"text": prompt}]
        for label, frames in clips:
            parts.append({"text": f"--- {label} ---"})
            for path in frames:
                parts.append({"inlineData": {"mimeType": "image/jpeg",
                                              "data": base64.b64encode(path.read_bytes()).decode("ascii")}})
        raw = self.client.generate(self.model, parts, max_tokens=self.max_tokens,
                                   timeout=600, attempts=4, temperature=0.0)
        return {"message": {"content": raw, "reasoning_content": None},
                "finish_reason": None, "usage": None, "max_tokens": self.max_tokens,
                "attempts": [], "error": None}


def prompt_for(question, qid, dataset):
    shown, back = shuffled_view(question["options"], qid)
    n = len(question["answer"])
    letters = sorted(shown)
    template = SINGLE_VIDEO_PROMPT if dataset.startswith("single_") else MULTI_VIDEO_PROMPT
    prompt = template.format(question=question["question"], options="\n".join(f"{letter}. {shown[letter]}" for letter in letters),
                           n=n, plural="s" if n > 1 else "", example=", ".join(f'"{letter}"' for letter in letters[:n]))
    return prompt, letters, back


def check_run_config(path, config):
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(f"Run configuration changed; choose another --run-id: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args, dataset, root, work):
    corpus_path, questions = load_questions(root, dataset)
    if args.limit:
        questions = questions[args.skip:args.skip + args.limit]
    elif args.skip:
        questions = questions[args.skip:]
    run_dir = work / "runs" / args.run_id
    prompt_template = SINGLE_VIDEO_PROMPT if dataset.startswith("single_") else MULTI_VIDEO_PROMPT
    config = {"version": VERSION, "dataset": dataset, "corpus_sha256": file_hash(corpus_path),
              "prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(), "model": args.model,
              "backend": args.backend,
              "thinking": args.thinking, "max_tokens": args.max_tokens,
              "max_tokens_cap": args.max_tokens_cap, "frames_per_question": args.frames,
              "min_frames_per_clip": args.min_frames, "max_width": args.max_width,
              "frame_extractor": "ffmpeg fps=k/duration; scale=min(max_width,iw):-2", "temperature": 0.0}
    if args.command == "run":
        check_run_config(run_dir / f"{dataset}.config.json", config)
    outfile = run_dir / f"{dataset}.jsonl"
    done = set()
    if outfile.is_file():
        for record in jsonl(outfile):
            if not record.get("error"):
                done.add(record["question_id"])
    questions = [q for q in questions if question_id(q, dataset) not in done]
    if args.command == "run" and args.backend == "openai" and not args.base_url:
        raise ValueError("--base-url is required for run")
    client = None
    if args.command == "run":
        if args.backend == "floodgate":
            if args.thinking != "auto":
                raise ValueError("Floodgate backend does not support --thinking true/false")
            client = FloodgateClient(root, args.model, args.max_tokens, args.rps)
        else:
            client = Client(args.base_url, args.model, args.api_key, args.thinking,
                            args.max_tokens, args.max_tokens_cap, args.rps)
    resolver = MediaResolver(root, Path(args.single_clips).expanduser(), work / "cache" / "media")
    prepared = []
    try:
        for question in questions:
            qid = question_id(question, dataset)
            try:
                clips = resolver.clips(dataset, question)
                frames, quotas = extract_question_frames(dataset, qid, clips, work / "cache" / "frames",
                                                         args.frames, args.min_frames, args.max_width)
                prepared.append((question, frames, quotas))
                if args.command == "prepare":
                    print(f"PREPARED {dataset} {qid} clips={len(frames)} frames={sum(len(f) for _, f in frames)} quotas={quotas}")
            except Exception as exc:
                prepared.append((question, exc, []))
                print(f"MEDIA_ERROR {dataset} {qid} {type(exc).__name__}: {exc}", flush=True)
    finally:
        resolver.close()
    if args.command == "prepare":
        print(f"PREPARE_DONE dataset={dataset} questions={len(questions)} ok={sum(not isinstance(f, Exception) for _, f, _ in prepared)}")
        return

    def evaluate(item):
        question, frames, quotas = item
        qid = question_id(question, dataset)
        if isinstance(frames, Exception):
            return {"question_id": qid, "dataset": dataset, "error": f"media:{type(frames).__name__}: {frames}"}
        prompt, letters, back = prompt_for(question, qid, dataset)
        started = time.time()
        try:
            response = client.ask(prompt, frames)
            message = response.get("message") or {}
            raw = message.get("content") or ""
            inline_reasoning, answer_text = split_reasoning(raw)
            reasoning = message.get("reasoning_content") or message.get("reasoning") or inline_reasoning
            thinking_required = args.thinking == "true" or "thinking" in args.model.lower()
            trace_present = bool(reasoning) or "Thinking Process:" in raw
            base = {"question_id": qid, "dataset": dataset, "unit": question.get("unit"),
                    "model": args.model, "thinking": args.thinking,
                    "raw_message": message, "raw": raw, "reasoning_content": reasoning,
                    "thinking_trace_required": thinking_required,
                    "thinking_trace_present": trace_present,
                    "finish_reason": response.get("finish_reason"), "usage": response.get("usage"),
                    "max_tokens_used": response.get("max_tokens"), "attempts": response.get("attempts"),
                    "latency_s": round(time.time() - started, 1)}
            if response.get("error"):
                return {**base, "error": response["error"]}
            if "<think>" in raw and "</think>" not in raw:
                return {**base, "error": "Unclosed thinking trace; final answer unavailable"}
            if thinking_required and not trace_present:
                return {**base, "error": "Thinking trace missing from model response"}
            displayed, parse_error = parse_answer(answer_text, letters)
            if parse_error:
                return {**base, "error": f"No parseable final answer: {parse_error}"}
            prediction = sorted({back[x] for x in displayed if x in back})
            gold = sorted(question["answer"])
            return {**base, "question_type": question.get("question_type"), "n_select": len(gold),
                    "n_options": len(question["options"]), "n_clips": len(frames),
                    "n_frames": sum(len(f) for _, f in frames), "frame_quotas": quotas,
                    "pred": prediction, "gold": gold,
                    "exact": prediction == gold, "overlap": len(set(prediction) & set(gold)) / max(len(gold), 1),
                    "n_pred": len(prediction)}
        except Exception as exc:
            return {"question_id": qid, "dataset": dataset, "model": args.model,
                    "thinking": args.thinking, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}

    counts = collections.Counter()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "a", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(evaluate, item) for item in prepared]):
            record = future.result()
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            counts["error" if record.get("error") else "ok"] += 1
            if sum(counts.values()) % 10 == 0:
                print(f"PROGRESS {dataset} {sum(counts.values())}/{len(prepared)} ok={counts['ok']} error={counts['error']}", flush=True)
    print(f"RUN_DONE dataset={dataset} ok={counts['ok']} error={counts['error']} output={outfile}")


def question_groups(questions, dataset):
    groups = {"overall": questions}
    if dataset == "mixed_1487":
        groups["multi_video_531"] = [q for q in questions if q.get("unit") in MULTI_UNITS]
    units = sorted({q.get("unit") for q in questions if q.get("unit")})
    if len(units) > 1:
        for unit in units:
            groups[unit] = [q for q in questions if q.get("unit") == unit]
    return groups


def summarize(members, results, dataset):
    """Score a question group. Primary metrics use every question as the denominator:
    failed (error) and missing (never run) questions count as wrong."""
    total = len(members)
    valid, failed = [], 0
    for q in members:
        row = results.get(question_id(q, dataset))
        if row is None:
            continue
        if not row.get("error") and "exact" in row:
            valid.append(row)
        else:
            failed += 1
    exact = sum(r["exact"] for r in valid)
    pct = lambda x, n: 100 * x / n if n else 0.0
    return {"total": total, "completed": len(valid), "failed": failed,
            "missing": total - len(valid) - failed, "exact": exact,
            "exact_pct": pct(exact, total), "exact_completed_pct": pct(exact, len(valid)),
            "overlap_pct": pct(sum(r["overlap"] for r in valid), total),
            "wellformed_pct": pct(sum(r["n_pred"] == r["n_select"] for r in valid), total)}


def load_results(root, work, run_id, dataset):
    """Return the latest record per question, warning if the corpus changed since the run."""
    run_dir = work / "runs" / run_id
    path = run_dir / f"{dataset}.jsonl"
    if not path.is_file():
        return None, None
    config_path = run_dir / f"{dataset}.config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    corpus_path = root / DATASETS[dataset]
    if config.get("corpus_sha256") and config["corpus_sha256"] != file_hash(corpus_path):
        print(f"WARNING {run_id} {dataset}: corpus_sha256 differs from current {corpus_path}", file=sys.stderr)
    return {row["question_id"]: row for row in jsonl(path)}, config


def score(root, work, run_id, dataset):
    _, questions = load_questions(root, dataset)
    results, _ = load_results(root, work, run_id, dataset)
    if results is None:
        print(f"NO_RESULTS {dataset} {work / 'runs' / run_id / f'{dataset}.jsonl'}")
        return
    overall = summarize(questions, results, dataset)
    print(f"SCORE {dataset} completed={overall['completed']}/{overall['total']} "
          f"failed={overall['failed']} missing={overall['missing']}")
    if overall["missing"]:
        print(f"  PARTIAL RUN: {overall['missing']} questions have no result; this is not a full-run score")
    for name, members in question_groups(questions, dataset).items():
        s = summarize(members, results, dataset)
        if s["total"]:
            print(f"  {name}: exact={s['exact']}/{s['total']} ({s['exact_pct']:.2f}%) "
                  f"[completed-only {s['exact']}/{s['completed']} ({s['exact_completed_pct']:.2f}%)] "
                  f"overlap={s['overlap_pct']:.2f}% wellformed={s['wellformed_pct']:.2f}% "
                  f"failed={s['failed']} missing={s['missing']}")


def compare(root, work, datasets):
    """Show coverage and scores across recorded model/mode runs (failures count as wrong)."""
    all_questions = {name: load_questions(root, name)[1] for name in datasets}
    print("run_id\tmodel\tthinking\tdataset\tgroup\ttotal\tcompleted\tfailed\tmissing\t"
          "exact\texact_pct\texact_completed_pct\toverlap_pct\twellformed_pct")
    for run_dir in sorted((work / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        for dataset in datasets:
            if not (run_dir / f"{dataset}.config.json").is_file():
                continue
            results, config = load_results(root, work, run_dir.name, dataset)
            if results is None:
                continue
            for group, members in question_groups(all_questions[dataset], dataset).items():
                s = summarize(members, results, dataset)
                print(f"{run_dir.name}\t{config['model']}\t{config['thinking']}\t{dataset}\t{group}\t"
                      f"{s['total']}\t{s['completed']}\t{s['failed']}\t{s['missing']}\t{s['exact']}\t"
                      f"{s['exact_pct']:.2f}\t{s['exact_completed_pct']:.2f}\t{s['overlap_pct']:.2f}\t"
                      f"{s['wellformed_pct']:.2f}")


def inventory(root, single_clips, datasets):
    for dataset in datasets:
        _, questions = load_questions(root, dataset)
        units = collections.Counter(q.get("unit") for q in questions)
        choices = collections.Counter(len(q["answer"]) for q in questions)
        print(f"{dataset}: questions={len(questions)} units={dict(units)} answer_count={dict(choices)}")
        if dataset.startswith("single_"):
            missing = [q["id"] for q in questions if not (single_clips / f"{q['id']}.mp4").is_file()]
            print(f"  media={len(questions)-len(missing)}/{len(questions)} missing={missing[:10]}")
        else:
            bundles = collections.Counter(q["bundle"] for q in questions)
            missing = [b for b in bundles if not (root / "annotation_bundles" / f"{b}.tar").is_file()]
            print(f"  bundles={dict(bundles)} missing={missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inventory", "prepare", "run", "score", "compare"])
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--root", default="~/datasets/RunningBench")
    parser.add_argument("--single-clips", default="~/datasets/seg600_verification/videoclips")
    parser.add_argument("--work-dir", default="~/datasets/RunningBench/unified_eval/work")
    parser.add_argument("--run-id", default="qwen3vl8b-instruct-v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--backend", choices=["openai", "floodgate"], default="openai")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--thinking", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens-cap", type=int, default=32768)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=480)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rps", type=float, default=0.4)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.max_tokens_cap < args.max_tokens:
        parser.error("--max-tokens-cap must be >= positive --max-tokens")
    root = Path(args.root).expanduser()
    work = Path(args.work_dir).expanduser()
    selected = ["single_698", "mixed_1487"] if args.dataset == "both" else [args.dataset]
    if args.command == "inventory":
        inventory(root, Path(args.single_clips).expanduser(), selected)
    elif args.command == "score":
        for dataset in selected:
            score(root, work, args.run_id, dataset)
    elif args.command == "compare":
        compare(root, work, selected)
    else:
        for dataset in selected:
            run(args, dataset, root, work)


if __name__ == "__main__":
    main()
