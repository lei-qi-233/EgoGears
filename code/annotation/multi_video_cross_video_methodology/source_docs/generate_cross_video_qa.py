#!/usr/bin/env python3
"""Cross-video questions for RunningBench scene groups.

For one scene group (a `route_id` from the route manifest) this script:
  1. cuts every member video into 3-minute windows and compresses each window on
     its own (≤11 MiB ⇒ ~500 kbps, instead of 50-60 kbps for a whole 15-minute video);
  2. has gemini-3.1-pro observe each window and return TIMESTAMPED landmarks,
     events and turns (videos are anonymised A/B/C; the model never sees filenames);
  3. merges the windows into one absolute-time timeline per video;
  4. has gemini-3.5-flash write questions that need ≥2 videos, each with option
     letters, an answer list and an evidence window for EVERY video involved;
  5. runs four gates — structure, leakage blacklist, visual re-check on the cited
     evidence clips only, and a video-blind guess — then shuffles the options.

Nothing here touches the existing annotations. Output:
  /mnt/data/data_anno/runningbench_cross_video_qa/<route_id>/
      windows/<video>/w<k>.api.mp4      compressed windows
      observations/<video>/w<k>.json    timestamped window observations
      timeline.json                     merged per-video timelines
      questions.raw.json                as generated
      questions.json                    after gates + shuffle

Usage:
  python3 generate_cross_video_qa.py --group route_rhein --dry-run
  python3 generate_cross_video_qa.py --group route_rhein --n-questions 8
"""
import argparse, base64, importlib.util, json, math, os, re, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = Path("/mnt/data/cvhci_video_understanding/metadata/route_manifest/route_manifest.json")
OUT_ROOT = Path("/mnt/data/data_anno/runningbench_cross_video_qa")
FFMPEG = Path("/mnt/data/data_anno/ffmpeg-7.0.2-amd64-static/ffmpeg")
WINDOW_S = 180
TARGET_SINGLE, TARGET_MULTI = 6, 8
QUOTA_PER_SECOND = 0.45
# gemini-3.5-flash spends "thinking" tokens out of maxOutputTokens; below ~16k the visible
# JSON gets starved (observed: 544 chars at 8192 vs a complete 8-question payload at 32768).
TYPES = ("route-phase-alignment", "landmark-visibility-comparison", "transient-event-attribution",
         "entry-and-merge-comparison", "reverse-view-recognition", "route-retrieval")
LEAK = ("fast", "slow", "normal speed", "trial", "traj", ".mov", ".mp4", "filename", "described", "the evidence",
        "observation", "the timeline", "annotation", "according to the data", "recorded on", "p01", "p02", "p03")

OBSERVE_PROMPT = """You are watching window {k} of anonymous first-person video {vid}. This window covers
{start}s to {end}s of the full recording; timestamps you report are relative to the START OF THIS WINDOW
as mm:ss. Do not infer anything from filenames — you have none.

Return one JSON object:
{{
 "landmarks": [ {{"name": "...", "first_seen": "mm:ss", "last_seen": "mm:ss", "side": "left|right|ahead|behind",
                 "detail_visible": "readable text / clear / blurred / distant", "notes": "..."}} ],
 "events":    [ {{"description": "...", "at": "mm:ss", "kind": "person|vehicle|animal|obstacle|other"}} ],
 "turns":     [ {{"kind": "left|right|u-turn|straight-on|stop", "at": "mm:ss", "cue": "what marks the turn"}} ],
 "surface_and_setting": "...",
 "visibility": "lighting, weather, blur, occlusion",
 "uncertain": ["things you could not verify"]
}}
Only report what is visible. Prefer distinctive, static landmarks (signs, structures, unusual objects)
that would let someone recognise this exact place in another recording."""

QUESTION_PROMPT = """You are writing benchmark questions that can ONLY be answered by watching and comparing
several first-person recordings of the same route. Anonymous videos: {vids}.

TIMELINES (absolute seconds from the start of each video; the only ground truth you have):
{timelines}

Write exactly {n} questions, at most 2 of any one type. Spread them across these types (at least one of each
that the evidence supports; skip a type only if the timelines genuinely cannot support it):
- route-phase-alignment: when does a landmark shared by two videos appear in EACH of them?
- landmark-visibility-comparison: in which video is a shared landmark clearly visible / readable, and where is it only blurred or distant?
- transient-event-attribution: which video contains a one-off event (person/vehicle/animal), and when?
- entry-and-merge-comparison: how do the videos differ in how they start or join the shared path?
- reverse-view-recognition: does any pair traverse the same stretch in opposite directions (a landmark on the left in one is on the right in the other)?
- route-retrieval: given an ORDERED sequence of 3 landmarks, which videos contain that sequence in that order?

Rules for every question:
- At least {n_multi} questions must have MORE THAN ONE correct answer (e.g. "select all videos that …").
- Options must be length-matched: every distractor within ±30% of the correct option's word count. Never let the
  correct option be the longest or the most detailed one.
- When options contain timestamps, all options must use the SAME set of time windows re-assigned/permuted across
  videos, so no option looks more "orderly" than the others. Prefer asking about ORDER or about which video is
  earliest/latest over asking for exact seconds.
- No two questions may hinge on the same landmark or the same event.
- The question stem must never contain a timestamp that identifies the correct option; ask "in which video…" or
  "which comes first…", never "…around the 440-second mark".
- For route-retrieval, every video that CONTAINS the sequence must get one evidence window PER LANDMARK (so a
  checker can confirm the order); videos that lack the sequence get one window showing where it would be.
- Refer to videos only by their letters ({vids}). Never mention speed, pace class, date, trial numbers or filenames.
- Ask as if the reader is WATCHING the videos; never say "described", "the timeline", "the observation".
- `options` is a JSON object keyed A.., exactly {single} options for single-answer questions and {multi} for
  multi-answer ones. Distractors must be plausible and definitively wrong given the timelines.
- `answer` is a non-empty list of option letters.
- `required_video_ids` lists every video the reader must watch (≥2).
- `evidence` gives, for EVERY video in required_video_ids, one or more [start_s, end_s] windows (≤40 s each)
  that a checker can watch to confirm the answer — including videos where the answer is "not present".

Return only: {{"questions": [ {{"question_type": "...", "question": "...", "options": {{...}}, "answer": [...],
"required_video_ids": [...], "evidence": {{"A": [[s,e]], ...}}, "why_hard": "..."}} ]}}"""

RECHECK_PROMPT = """You are verifying one benchmark question against the cited video evidence. The clips below
are cut from anonymous videos {vids} exactly at the cited moments (clip labels give video letter and
absolute time span).

QUESTION: {question}
OPTIONS: {options}
PROPOSED ANSWER: {answer}

Watch the clips. Return only JSON: {{"verdict": "supported" | "contradicted" | "insufficient",
"correct_answer_if_different": [...] or null, "notes": "what you saw"}}"""

BLIND_PROMPT = """Answer this multiple-choice question WITHOUT any video — you have none. Pick the most likely
option letters. Return only JSON: {{"answer": ["..."]}}

QUESTION: {question}
OPTIONS: {options}
{arity}"""


class RateLimiter:
    def __init__(self, per_second):
        self.interval, self.lock, self.next_slot = 1.0 / per_second, threading.Lock(), time.monotonic()

    def acquire(self):
        with self.lock:
            now = time.monotonic(); wait = max(0.0, self.next_slot - now)
            self.next_slot = max(now, self.next_slot) + self.interval
        if wait:
            time.sleep(wait)


class PacedSession:
    def __init__(self, session, limiter):
        self._s, self._l = session, limiter

    def post(self, *a, **k):
        self._l.acquire(); return self._s.post(*a, **k)

    def __getattr__(self, n):
        return getattr(self._s, n)


def load_generator():
    spec = importlib.util.spec_from_file_location("repair_runningbench_annotations",
                                                  HERE / "repair_runningbench_annotations.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m


def mmss(s: str) -> float:
    parts = [float(x) for x in str(s).strip().split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0]


def validate_observation(d):
    for k in ("landmarks", "events", "turns"):
        if not isinstance(d.get(k), list):
            raise ValueError(f"{k} must be a list")
    for lm in d["landmarks"]:
        mmss(lm["first_seen"]); mmss(lm["last_seen"])
    for ev in d["events"]:
        mmss(ev["at"])
    for t in d["turns"]:
        mmss(t["at"])


def validate_blind(d):
    if not isinstance(d.get("answer"), list) or not d["answer"]:
        raise ValueError("answer must be a non-empty list of letters")


def make_question_validator(vids, n):
    def validate(d):
        qs = d.get("questions")
        if not isinstance(qs, list) or len(qs) != n:
            raise ValueError(f"need exactly {n} questions")
        for q in qs:
            if q.get("question_type") not in TYPES:
                raise ValueError("bad question_type")
            opts = q.get("options")
            if isinstance(opts, list):
                q["options"] = opts = {chr(65 + i): o for i, o in enumerate(opts)}
            if not isinstance(opts, dict) or sorted(opts) != [chr(65 + i) for i in range(len(opts))]:
                raise ValueError("options must be A.. consecutive")
            ans = q.get("answer")
            if not isinstance(ans, list) or not ans or not set(ans) <= set(opts):
                raise ValueError("answer must be non-empty subset of options")
            want = TARGET_MULTI if len(ans) > 1 else TARGET_SINGLE
            if len(opts) != want:
                raise ValueError(f"expected {want} options, got {len(opts)}")
            req = q.get("required_video_ids")
            if not isinstance(req, list) or len(req) < 2 or not set(req) <= set(vids):
                raise ValueError("required_video_ids must name ≥2 of the videos")
            ev = q.get("evidence")
            if not isinstance(ev, dict) or any(v not in ev or not ev[v] for v in req):
                raise ValueError("evidence missing for a required video")
            for v, spans in ev.items():
                for s in spans:
                    if not (isinstance(s, list) and len(s) == 2 and 0 <= s[0] < s[1] and s[1] - s[0] <= 60):
                        raise ValueError(f"bad evidence span {s} for {v}")
            low = (q["question"] + " " + " ".join(opts.values())).lower()
            hit = [w for w in LEAK if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low)]
            if hit:
                raise ValueError(f"leak: {hit[0]!r}")
            # A timestamp that appears both in the stem and in a correct option (but in no
            # distractor) answers the question for anyone who can read numbers.
            stem_nums = set(re.findall(r"(?<!\d)(\d{2,4})(?!\d)", q["question"]))
            if stem_nums:
                correct_nums = set(n for k in ans for n in re.findall(r"(?<!\d)(\d{2,4})(?!\d)", opts[k]))
                give = stem_nums & correct_nums
                if give:
                    raise ValueError(f"stem leaks the answer timestamp {sorted(give)}")
        if sum(1 for i in qs if len(i["answer"]) > 1) < max(2, n // 3):
            raise ValueError(f"need at least {max(2, n // 3)} multi-answer questions")
        per_type = {}
        for i in qs:
            per_type[i["question_type"]] = per_type.get(i["question_type"], 0) + 1
        over = [t for t, c in per_type.items() if c > max(2, n // 4)]
        if over:
            raise ValueError(f"too many questions of type {over[0]} (max {max(2, n // 4)} per type)")
        for q in qs:
            words = {k: len(v.split()) for k, v in q["options"].items()}
            longest = max(words.values())
            if all(words[k] == longest for k in q["answer"]) and \
                    sum(1 for w in words.values() if w == longest) == len(q["answer"]):
                raise ValueError("the correct option(s) are the longest — length cue leaks the answer")
            if q["question_type"] == "route-retrieval":
                # Videos that lack the sequence cannot cite one window per landmark, so require
                # multi-window evidence only for whichever video(s) do contain it.
                if max(len(q["evidence"].get(v, [])) for v in q["required_video_ids"]) < 2:
                    raise ValueError("route-retrieval needs one evidence window per landmark for the matching video(s)")
    return validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--n-questions", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-recheck", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="cut and compress windows; no API calls")
    ap.add_argument("--observe-only", action="store_true", help="build the 3-minute window time index only; no questions")
    a = ap.parse_args()

    gen = load_generator()
    man = json.load(open(MANIFEST))
    members = [r for r in man["videos"] if r["route_id"] == a.group and not r["duplicate_of"]]
    if len(members) < 2:
        raise SystemExit(f"group {a.group} has {len(members)} usable videos")
    members.sort(key=lambda r: r["logical_video_id"])
    letters = {chr(65 + i): r for i, r in enumerate(members)}
    out = OUT_ROOT / a.group; out.mkdir(parents=True, exist_ok=True)

    plan = []
    for L, r in letters.items():
        D = r["duration_s"]; n = math.ceil(D / WINDOW_S)
        # A trailing window shorter than 30 s is useless on its own; fold it into the previous one.
        if n > 1 and D - (n - 1) * WINDOW_S < 30:
            n -= 1
        for k in range(n):
            s = k * WINDOW_S; e = D if k == n - 1 else s + WINDOW_S
            plan.append((L, r, k, s, e))
    print(f"group {a.group}: {len(members)} videos → {len(plan)} windows of {WINDOW_S}s")
    for L, r in letters.items():
        print(f"  {L} = {r['logical_video_id']}  {r['duration_s']:.0f}s  shape={r['shape']}  dir={r['direction']}")
    json.dump({L: r["logical_video_id"] for L, r in letters.items()}, open(out / "anonymous_map.json", "w"), indent=1)
    if a.dry_run:
        print(f"video calls: {len(plan)} | text calls: ~{1 + 2 * a.n_questions} | recheck video calls: {a.n_questions}")
        return 0
    if a.prepare_only:
        for L, r, k, s, e in plan:
            clip = out / "windows" / r["logical_video_id"] / f"w{k}.mp4"
            clip.parent.mkdir(parents=True, exist_ok=True)
            if not clip.exists():
                gen.make_segment(FFMPEG, Path(r["file"]), clip, s, e - s)
            ac = gen.api_copy(FFMPEG, clip, clip.parent)
            print(f"  prepared {L} w{k} {s:.0f}-{e:.0f}s → {ac.name} {ac.stat().st_size / 1e6:.1f} MB", flush=True)
        return 0

    token = os.environ.get("FLOODGATE_PROJECT_TOKEN")
    if not token:
        raise SystemExit("FLOODGATE_PROJECT_TOKEN is required")
    api = gen.Floodgate(token); api.session = PacedSession(api.session, RateLimiter(QUOTA_PER_SECOND))

    # ---- 1-2: windows + timestamped observations ----
    for L, r, k, s, e in plan:
        obs_path = out / "observations" / r["logical_video_id"] / f"w{k}.json"
        if obs_path.exists():
            continue
        clip = out / "windows" / r["logical_video_id"] / f"w{k}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        if not clip.exists():
            gen.make_segment(FFMPEG, Path(r["file"]), clip, s, e - s)
        api_clip = gen.api_copy(FFMPEG, clip, clip.parent)
        enc = base64.b64encode(api_clip.read_bytes()).decode("ascii")
        for attempt in range(4):
            try:
                raw = api.generate(gen.CAPTION_MODEL,
                                   [{"text": OBSERVE_PROMPT.format(k=k, vid=L, start=int(s), end=int(e))},
                                    {"inlineData": {"mimeType": "video/mp4", "data": enc}}], 8192, json_mode=True)
                obs = gen.clean_json(raw); validate_observation(obs); break
            except Exception as exc:
                print(f"  observe retry {attempt + 1}/4 {L} w{k}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                obs = None
        if obs is None:
            print(f"  !! window {L} w{k} failed; continuing", flush=True); continue
        gen.write_json(obs_path, {"video": L, "window": k, "window_start_s": s, "window_end_s": e,
                                  "api_clip_bytes": api_clip.stat().st_size, "observation": obs})
        print(f"  observed {L} w{k} ({s:.0f}-{e:.0f}s, {api_clip.stat().st_size / 1e6:.1f} MB)", flush=True)

    if a.observe_only:
        n_obs = sum(1 for _ in (out / "observations").glob("*/w*.json")) if (out / "observations").exists() else 0
        print(f"observe-only: {n_obs} window observations on disk for {a.group}", flush=True)
        return 0

    # ---- 3: merge into absolute timelines ----
    timelines = {}
    for L, r in letters.items():
        tl = {"landmarks": [], "events": [], "turns": [], "visibility": [], "setting": []}
        for p in sorted((out / "observations" / r["logical_video_id"]).glob("w*.json"),
                        key=lambda p: int(p.stem[1:])):
            w = json.load(open(p)); o = w["observation"]; base = w["window_start_s"]
            for lm in o["landmarks"]:
                tl["landmarks"].append({**lm, "first_seen_s": round(base + mmss(lm["first_seen"])),
                                        "last_seen_s": round(base + mmss(lm["last_seen"]))})
            for ev in o["events"]:
                tl["events"].append({**ev, "at_s": round(base + mmss(ev["at"]))})
            for t in o["turns"]:
                tl["turns"].append({**t, "at_s": round(base + mmss(t["at"]))})
            tl["visibility"].append(f"{base:.0f}s: {o.get('visibility', '')}")
            tl["setting"].append(f"{base:.0f}s: {o.get('surface_and_setting', '')}")
        for lst in ("landmarks", "events", "turns"):
            for item in tl[lst]:
                for key in ("first_seen", "last_seen", "at"):
                    item.pop(key, None)
        tl["duration_s"] = round(r["duration_s"])
        timelines[L] = tl
    json.dump(timelines, open(out / "timeline.json", "w"), indent=1, ensure_ascii=False)
    print(f"timelines: " + ", ".join(f"{L}:{len(t['landmarks'])}lm/{len(t['events'])}ev/{len(t['turns'])}turns"
                                    for L, t in timelines.items()), flush=True)

    # ---- 4: questions ----
    vids = ", ".join(letters)
    qs = gen.generate_valid_json(api, QUESTION_PROMPT.format(
        vids=vids, timelines=json.dumps(timelines, ensure_ascii=False), n=a.n_questions,
        single=TARGET_SINGLE, multi=TARGET_MULTI, n_multi=max(2, a.n_questions // 3)),
        make_question_validator(list(letters), a.n_questions), 32768)["questions"]
    json.dump(qs, open(out / "questions.raw.json", "w"), indent=1, ensure_ascii=False)
    print(f"generated {len(qs)} questions", flush=True)

    # ---- 5: gates ----
    for i, q in enumerate(qs, 1):
        # Shuffle FIRST: the generator writes the correct option in position A, and a
        # blind guesser also favours A, so gating in generated order overstates guessability.
        gen.shuffle_options([q])
        q["option_order"] = "shuffled"
        q["gates"] = {"structure": "pass", "leak_scan": "pass"}
        # blind guess (video-blind, options only)
        try:
            arity = "Exactly one option is correct." if len(q["answer"]) == 1 else "More than one option is correct."
            hits, guesses = 0, []
            for trial in range(3):   # one sample is a coin flip at 6 options; take a majority of three
                bg = gen.generate_valid_json(api, BLIND_PROMPT.format(
                    question=q["question"], options=json.dumps(q["options"], ensure_ascii=False),
                    arity=arity + f" (attempt {trial + 1})"), validate_blind, 4096)
                guesses.append(sorted(bg["answer"]))
                hits += sorted(bg["answer"]) == sorted(q["answer"])
            q["gates"]["blind_guess"] = {"guesses": guesses, "hits_of_3": hits, "matches_gold": hits >= 2}
        except Exception as exc:
            q["gates"]["blind_guess"] = {"error": type(exc).__name__}
        # visual recheck on cited evidence only
        if a.skip_recheck:
            q["gates"]["visual_recheck"] = "skipped"
        else:
            parts = [{"text": RECHECK_PROMPT.format(vids=vids, question=q["question"],
                                                    options=json.dumps(q["options"], ensure_ascii=False),
                                                    answer=q["answer"])}]
            total = 0
            # Videos named in the correct option(s) first, so a size cap never drops the
            # clip that decides the verdict (pilot 3: C was cut off → "insufficient").
            named = [L for L in q["required_video_ids"] if any(f"video {L}".lower() in q["options"][k].lower()
                                                                or f" {L}," in q["options"][k] or q["options"][k].startswith(f"{L} ")
                                                                for k in q["answer"])]
            order = named + [L for L in q["required_video_ids"] if L not in named]
            for L in order:
                r = letters[L]
                for (s, e) in q["evidence"][L][:3]:
                    s2, e2 = max(0, s - 5), min(r["duration_s"], e + 5)
                    clip = out / "recheck" / f"q{i}_{L}_{int(s2)}_{int(e2)}.480p.mp4"; clip.parent.mkdir(exist_ok=True)
                    if not clip.exists():
                        raw_clip = clip.with_suffix(".src.mp4")
                        gen.make_segment(FFMPEG, Path(r["file"]), raw_clip, s2, e2 - s2)
                        import subprocess
                        # Never re-check at lower quality than the observation pass: pilot 4 judged
                        # signs "blurred in all three" from 360p clips that were legible at 480p.
                        subprocess.run([str(FFMPEG), "-y", "-i", str(raw_clip), "-vf", "scale=-2:480", "-c:v", "libx264",
                                        "-preset", "veryfast", "-crf", "30", "-an", "-movflags", "+faststart", str(clip)],
                                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        raw_clip.unlink(missing_ok=True)
                    size = clip.stat().st_size
                    if total + size > 20 * 1024 * 1024:
                        q.setdefault("recheck_dropped_clips", []).append(f"{L}:{int(s2)}-{int(e2)}")
                        continue
                    total += size
                    parts.append({"text": f"[clip: video {L}, {int(s2)}-{int(e2)}s]"})
                    parts.append({"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(clip.read_bytes()).decode("ascii")}})
            try:
                raw = api.generate(gen.CAPTION_MODEL, parts, 8192, json_mode=True)
                q["gates"]["visual_recheck"] = gen.clean_json(raw)
            except Exception as exc:
                q["gates"]["visual_recheck"] = {"verdict": "error", "notes": type(exc).__name__}
        v = q["gates"]["visual_recheck"]
        print(f"  q{i} [{q['question_type']}] recheck={v if isinstance(v, str) else v.get('verdict')} "
              f"blind={q['gates']['blind_guess'].get('matches_gold')}", flush=True)

    json.dump({"group": a.group, "anonymous_map": {L: r["logical_video_id"] for L, r in letters.items()},
               "questions": qs}, open(out / "questions.json", "w"), indent=1, ensure_ascii=False)
    kept = [q for q in qs if isinstance(q["gates"]["visual_recheck"], dict)
            and q["gates"]["visual_recheck"].get("verdict") == "supported"]
    print(f"\nwrote {out / 'questions.json'}: {len(qs)} generated, {len(kept)} pass visual recheck, "
          f"{sum(1 for q in qs if q['gates']['blind_guess'].get('matches_gold'))} guessable blind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
