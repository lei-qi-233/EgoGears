#!/usr/bin/env python3
"""Full-video annotations and questions for RunningBench.

The evaluation unit is the COMPLETE original video (one run). Nothing is sliced into
samples; 60 s / 180 s windows survive only as an internal time index, and every answer
carries evidence spans that are used for verification, never as data.

Per route (scene_id → route_id → run_id from scene_manifest.json):

  1. GLOBAL PASS   whole video, 360p/10 fps (≤11 MiB, else 288p) → gemini-3.1-pro →
                   route_phases, turns, return_to_start, revisited landmarks, environment
                   stages, dynamic events. Coarse but complete: order and structure.
  2. TIME INDEX    existing 3-minute window observations (if any) merged in as
                   `time_index` — finer timestamps for evidence; never a sample.
  3. WHOLE-VIDEO QUESTIONS  per run, from its own annotation only.
  4. CROSS-VIDEO QUESTIONS  per route, from all runs' annotations.
  5. GATES         structure · leak blacklist · stem-timestamp · length cue ·
                   HIGH-RES local re-check on the evidence spans (480p clips, so the
                   detail pass is sharper than the global pass) · blind guess ×3 ·
                   deterministic option shuffle.

Output: /mnt/data/data_anno/runningbench_fullvideo/<route_id>/
    global_input/<run>.360p.mp4          model input (not a sample)
    annotations/<run>.json               schema below
    questions/<run>.whole_video.json
    questions/<route>.cross_video.json

Usage:
  python3 build_fullvideo_annotations.py --route route_rhein --dry-run
  python3 build_fullvideo_annotations.py --route route_rhein [--n-whole 6] [--n-cross 8]
"""
import argparse, base64, importlib.util, json, os, re, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Overridable so a side corpus (e.g. duration-controlled excerpts) can run the same
# pipeline without writing into the main one.
SCENES = Path(os.environ.get("RB_SCENES",
    "/mnt/data/cvhci_video_understanding/metadata/route_manifest/scene_manifest.json"))
OUT_ROOT = Path(os.environ.get("RB_OUT_ROOT", "/mnt/data/data_anno/runningbench_fullvideo"))
WINDOW_ROOT = Path(os.environ.get("RB_WINDOW_ROOT",
    "/mnt/data/data_anno/runningbench_cross_video_qa"))                   # 3-min observations
FFMPEG = Path("/mnt/data/data_anno/ffmpeg-7.0.2-amd64-static/ffmpeg")
INLINE_CAP = 11 * 1024 * 1024
TARGET_SINGLE, TARGET_MULTI = 6, 8
QUOTA_PER_SECOND = 0.45

WHOLE_TYPES = ("route-summary", "start-end-relation", "landmark-order", "multi-step-turns",
               "environment-stages", "landmark-revisit", "event-phase")
# "same-route-identification" (1/9 usable) and "route-retrieval" (2/20) were dropped in v2.
# Both asked for a whole-video universal — "are these the same route", "which videos contain
# this sequence" — which no set of ≤60 s windows can settle, so the re-checker kept finding a
# fork or a match outside the cited evidence and ruling the gold answer contradicted. Their
# replacements ask WHERE the two videos part and WHICH ORDER a pair of landmarks comes in:
# same skill, but the answer is decidable from the windows themselves.
CROSS_TYPES = ("route-divergence-location", "divergence-point", "reverse-view-recognition",
               "unique-transient-obstacle", "different-start-shared-path", "route-phase-alignment",
               "landmark-visibility-comparison", "landmark-order-discrimination", "description-matching")
# Whole-video claims ("throughout", "at any point", "the same route") are what the checker
# cannot settle from windows, so the prompt bans them. They are NOT gated here: backtesting a
# phrase blacklist over the 180 v1 cross-video questions caught only 4, and 2 of those were
# questions that had passed both gates. Shape generation, do not reject on string match.
LEAK = ("fast", "slow", "normal speed", "trial", "traj", ".mov", ".mp4", "filename", "described", "the evidence",
        "observation", "the timeline", "annotation", "according to the data", "recorded on", "p01", "p02", "p03",
        "the annotation", "the summary")

GLOBAL_PROMPT = """You are watching {part_desc} of a first-person walking/running recording, anonymous id {vid},
{dur} seconds long in total, shown at reduced resolution. Do not infer anything from a filename — you have none.
A clock in the TOP-LEFT corner shows the true recording time as MM:SS. READ every time you report from that
clock and convert to seconds; never estimate time from pacing. (This part covers {p0}–{p1} s.)
Your job is the STRUCTURE of the route, in order. Fine details are checked separately.

Return one JSON object:
{{
 "route_phases": [ {{"start_sec": 0, "end_sec": 145, "description": "...", "landmarks": ["...", "..."],
                    "surface": "asphalt|gravel|paved|dirt|mixed", "setting": "..."}} ],
 "turns": [ {{"time_sec": 212, "direction": "left|right|u-turn|straight-on", "evidence": "what marks it"}} ],
 "start_end_relation": {{"returns_near_start": true|false|"unclear", "evidence": "..."}},
 "revisited_landmarks": [ {{"landmark": "...", "first_sec": 0, "second_sec": 0, "direction_change": "same|opposite|unclear"}} ],
 "environment_stages": ["..."],
 "dynamic_events": [ {{"time_sec": 0, "description": "...", "kind": "person|vehicle|animal|obstacle|other"}} ],
 "uncertainties": ["..."]
}}
Phases must tile {p0}–{p1} s in order (each end_sec = next start_sec; first start_sec = {p0}; last end_sec = {p1}).
Use only what is visible. Prefer distinctive static landmarks that identify this exact place."""

WHOLE_Q_PROMPT = """Write exactly {n} benchmark questions about ONE complete first-person video ({dur} s), anonymous id {vid}.
The reader WATCHES THE WHOLE VIDEO; questions must need the whole route, not one moment.

ANNOTATION (the only ground truth):
{annotation}

SIBLING RUNS on the same route (use their phases/landmarks as in-scene distractor material):
{siblings}

Cover these types, at most 2 per type:
- route-summary: what sequence of environments/phases does the route pass through?
- start-end-relation: does the walker end near where they started; what shows it?
- landmark-order: which landmark comes before/after another?
- multi-step-turns: what sequence of turns happens between two landmarks?
- environment-stages: where does the surface/setting change?
- landmark-revisit: is any landmark passed twice, and from which direction the second time?
- event-phase: in which route phase does a dynamic event occur?

Rules:
- DISTRACTORS MUST BE IN-SCENE. Build every wrong option from THIS video's own phases, landmarks and turns —
  re-ordered, mirrored (left↔right), shifted to the wrong phase, or paired with the wrong landmark. Sibling-run
  material may be used ONLY for landmarks/events that this video demonstrably lacks; for route-summary,
  environment-stages and start-end-relation NEVER use a sibling's description as a distractor — siblings walk the
  same route, so their true description is usually true here too. Use wrong-ORDER permutations of this video's own
  phases instead. Never introduce environments or objects that do not appear anywhere in the annotation
  (no beaches, highways, forests, indoor tracks, bridges, boardwalks, figure-eights unless they are actually there).
  A reader with no video and only general knowledge must find every option equally plausible.
- `start_end_relation.gpx_truth` and `facts.route_geometry`, when present, are ground truth from GPS; questions about
  returning to the start must agree with them and ask for the VISUAL evidence, not the fact.
- OUT-AND-BACK routes (facts.turnaround_sec present): every ORDER or LEFT/RIGHT question must name one leg ("on the
  outbound leg, before turning back" / "on the return leg") and ALL its evidence_spans must lie on that side of
  facts.turnaround_sec — otherwise it is ambiguous (left outbound = right on return).
- landmark-revisit questions need evidence_spans for BOTH passes (one before and one after facts.turnaround_sec).
- route-summary: never ask for the whole sequence of phases (its natural order is guessable from common sense);
  ask ONE local transition instead ("which setting comes immediately after X?").
- All options must be textually distinct from each other.
- At least {n_multi} questions have MORE THAN ONE correct option.
- Distractors length-matched (±30% words); the correct option is never the longest.
- The stem must not contain a timestamp that identifies the answer.
- `options` is an object keyed A.. with {single} options (one answer) or {multi} (several).
- `evidence_spans`: list of [start_sec, end_sec] windows a checker can watch to confirm the answer;
  each ≤60 s; give one span per landmark/turn the answer depends on. The `time_index` entries carry the
  PRECISE times — take evidence times from them when they exist (route_phases/dynamic_events times are
  approximate, drift up to ±60 s); pad each span by ±20 s.
- Never mention pace, date, trial numbers, filenames, or the words described/annotation/timeline.

Return only: {{"questions": [ {{"question_type": "...", "question": "...", "options": {{...}}, "answer": [...],
"evidence_spans": [[s,e], ...], "why_hard": "..."}} ]}}"""

CROSS_Q_PROMPT = """Write exactly {n} benchmark questions that can ONLY be answered by watching and comparing several
complete first-person recordings from the same scene. Anonymous videos: {vids}.

PER-VIDEO ANNOTATIONS (the only ground truth; times are seconds from each video's start):
{annotations}

Types (at most 2 per type; skip a type only if the evidence cannot support it):
- route-divergence-location: two videos share the route up to a point — which landmark is the LAST
  one both pass before their paths differ? (ask for the place, never for a yes/no "same route?")
- divergence-point: which two videos share the route up to a fork and differ after it — and at which landmark?
- reverse-view-recognition: which video passes a landmark from the opposite direction (left↔right)?
- unique-transient-obstacle: which recording has a one-off obstacle/vehicle/person, and in which phase?
- different-start-shared-path: which video starts elsewhere but joins the shared path, and where?
- route-phase-alignment: in which order do the videos reach a shared landmark (earliest → latest by video time)?
- landmark-visibility-comparison: in which video is a shared landmark close and legible vs distant/blurred?
- landmark-order-discrimination: two videos both pass landmarks X and Y — in which video does Y
  come first? (ask which ORDER, never which video "contains" a sequence)
- description-matching: match short route descriptions to the videos they describe.

Rules:
- DECIDABLE FROM THE WINDOWS: the answer must follow from the evidence spans you cite and nothing
  else. Never ask whether something happens "throughout", "at any point", "never", "only in",
  "from start to finish", or whether two routes are "the same" — settling those needs the whole
  video, the checker only gets your windows, and it will rule the question contradicted.
- Refer to videos only by their letters. Never mention pace class, date, trial numbers or filenames.
- At least {n_multi} questions have MORE THAN ONE correct option.
- Distractors length-matched (±30%); the correct option is never the longest.
- Timestamp options, if any, must be permutations of one set of values; prefer asking about ORDER.
- The stem must not contain a timestamp that identifies the answer.
- No two questions hinge on the same landmark or event.
- `options` keyed A.. with {single} (one answer) or {multi} (several) options.
- If the route has ONLY TWO videos, never write a single-answer question that reduces to "A or B" (50% blind
  odds). Use forms whose answer is not a video letter: order/alignment ("which landmark do both pass first?"),
  both/neither/multi-select over landmarks or events, side-swap statements, or "at what point do they diverge".
- `required_video_ids` ≥ 2. `evidence`: for EVERY required video, [start_sec, end_sec] windows (≤60 s) a checker
  can watch — one per landmark for videos that contain a sequence; one window for videos where it is absent.
  Take evidence times from `time_index` entries when they exist (they are precise); phase/event times in the
  annotation are approximate (drift up to ±60 s). Pad each span by ±20 s.

Return only: {{"questions": [ {{"question_type": "...", "question": "...", "options": {{...}}, "answer": [...],
"required_video_ids": [...], "evidence": {{"A": [[s,e]], ...}}, "why_hard": "..."}} ]}}"""

RECHECK_PROMPT = """Verify one benchmark question against high-resolution clips cut exactly at the cited evidence
(labels give the video letter and absolute seconds).

QUESTION: {question}
OPTIONS: {options}
PROPOSED ANSWER: {answer}

Return only JSON: {{"verdict": "supported" | "contradicted" | "insufficient",
"correct_answer_if_different": [...] or null, "notes": "what you saw"}}"""

BLIND_PROMPT = """Answer this multiple-choice question WITHOUT any video — you have none. Return only
JSON: {{"answer": ["..."]}}

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
    def __init__(self, session, limiter): self._s, self._l = session, limiter
    def post(self, *a, **k): self._l.acquire(); return self._s.post(*a, **k)
    def __getattr__(self, n): return getattr(self._s, n)


def load_generator():
    spec = importlib.util.spec_from_file_location("repair_runningbench_annotations", HERE / "repair_runningbench_annotations.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m


def ffmpeg(*args):
    subprocess.run([str(FFMPEG), "-v", "error", "-y", *map(str, args)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


GLOBAL_SPLIT_S = 900   # above this, one inline file would drop below ~70 kbps; use two input halves


def _clock_frames(tmpdir, start, length):
    """One PNG per second showing the true recording time — the global pass READS the clock
    instead of estimating it (estimates drifted +240 s by the end of a 448 s video)."""
    from PIL import Image, ImageDraw, ImageFont
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:
        font = ImageFont.load_default()
    for k in range(int(length) + 2):
        t = int(start) + k
        label = f"{t // 60:02d}:{t % 60:02d}"
        img = Image.new("RGB", (118, 36), (0, 0, 0))
        ImageDraw.Draw(img).text((8, 4), label, fill=(255, 255, 255), font=font)
        img.save(tmpdir / f"tc_{k:05d}.png")
    return tmpdir / "tc_%05d.png"


def _encode_budget(src, target, start, length):
    """Encode [start, start+length) of src into target so it fits the inline cap,
    with a burned-in MM:SS clock (top-left) on the full recording's time base."""
    import shutil, tempfile
    budget_kbps = int(INLINE_CAP * 8 * 0.90 / length / 1000)
    tmp = Path(tempfile.mkdtemp(prefix="tc_", dir=target.parent))
    try:
        pattern = _clock_frames(tmp, start, length)
        for h, kbps in ((288, budget_kbps), (240, int(budget_kbps * 0.95)), (216, int(budget_kbps * 0.9))):
            ffmpeg("-ss", start, "-t", length, "-i", src, "-framerate", "1", "-i", pattern,
                   "-filter_complex", f"[0:v]scale=-2:{h},fps=8[v];[v][1:v]overlay=6:6:shortest=1",
                   "-c:v", "libx264", "-preset", "veryfast", "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.15)}k",
                   "-bufsize", f"{kbps * 2}k", "-an", "-movflags", "+faststart", target)
            if target.stat().st_size <= INLINE_CAP:
                return target
        raise RuntimeError(f"cannot fit {target.name} under the inline cap ({target.stat().st_size} B)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def global_inputs(run, out):
    """Model input for the GLOBAL pass: [(path, start_sec, end_sec)], sized by bitrate budget.

    crf cannot promise a size (360p/crf30 came out at 29-38 MB for 7-10 min); a bitrate
    budget derived from duration can. 288p / 8 fps is enough for route structure — order,
    turns, phases — and every detail question is re-checked on 480p local clips anyway.
    Videos longer than GLOBAL_SPLIT_S are fed as two halves so neither drops below ~70 kbps;
    the halves are INPUT CHUNKS only — phases are merged back onto the full-video clock.
    """
    d = run["duration_sec"]
    (out / "global_input").mkdir(parents=True, exist_ok=True)
    pieces = [(0.0, d)] if d <= GLOBAL_SPLIT_S else [(0.0, d / 2), (d / 2, d)]
    result = []
    for k, (s0, s1) in enumerate(pieces):
        tag = "global" if len(pieces) == 1 else f"global.part{k + 1}of{len(pieces)}"
        target = out / "global_input" / f"{run['run_id']}.{tag}.mp4"
        if not (target.exists() and target.stat().st_size <= INLINE_CAP):
            _encode_budget(run["file"], target, s0, s1 - s0)
        result.append((target, s0, s1))
    return result


MERGE_PROMPT = """Two halves of ONE first-person recording ({dur} s) were annotated separately. Reconcile them.
Half 1 covers 0–{mid} s, half 2 covers {mid}–{dur} s. Known from GPS: {gpx_fact}

HALF 1 phases/turns:
{h1}

HALF 2 phases/turns:
{h2}

Return only JSON:
{{"revisited_landmarks": [{{"landmark": "...", "first_time_sec": s, "second_time_sec": s,
                           "second_pass_direction": "same|opposite", "evidence": "why these are the same place"}}],
  "turnaround": {{"time_sec": s or null, "evidence": "..."}},
  "start_end_relation": {{"returns_near_start": true|false, "evidence": "compare half-1 start with half-2 end"}} }}
Match landmarks by appearance and relative position, allowing left/right to swap on a return leg."""


def validate_merge(d):
    for k in ("revisited_landmarks", "turnaround", "start_end_relation"):
        if k not in d:
            raise ValueError(f"merge missing {k}")
    if not isinstance(d["revisited_landmarks"], list):
        raise ValueError("revisited_landmarks must be a list")


def merge_halves(api, gen, r, merged, mid, tag):
    """Reconcile two independently annotated halves: revisits, turnaround, start/end. Text-only."""
    shape = r.get("shape"); ta = r.get("turnaround_sec")
    gpx_fact = {"out_and_back": f"the route reverses at about {ta or mid} s and returns along the same path (landmarks re-passed, sides swapped)",
                "loop": "the route is a circuit returning near the start without retracing",
                "one_way": "the route does NOT return near the start"}.get(shape, "no GPS shape available")
    def digest(k):
        return json.dumps({"phases": [f"{p['start_sec']:.0f}-{p['end_sec']:.0f}s {p['description']} | {', '.join(p.get('landmarks', [])[:6])}"
                                      for p in merged["route_phases"] if (p["start_sec"] < mid) == (k == 1)],
                           "turns": [f"{t['time_sec']:.0f}s {t['direction']}" for t in merged["turns"] if (t["time_sec"] < mid) == (k == 1)]},
                          ensure_ascii=False)
    mg = gen.generate_valid_json(api, MERGE_PROMPT.format(dur=int(r["duration_sec"]), mid=mid, gpx_fact=gpx_fact,
                                                          h1=digest(1), h2=digest(2)), validate_merge, 16384)
    merged["revisited_landmarks"] = mg["revisited_landmarks"]
    merged["start_end_relation"] = dict(mg["start_end_relation"], source="merge of both halves + GPS")
    if mg["turnaround"].get("time_sec") is not None and not any("u" in t["direction"].lower() for t in merged["turns"]):
        merged["turns"].append({"time_sec": mg["turnaround"]["time_sec"], "direction": "u-turn",
                                "evidence": mg["turnaround"].get("evidence", "merge step")})
        merged["turns"].sort(key=lambda t: t["time_sec"])
    print(f"  merge {tag}: {len(mg['revisited_landmarks'])} revisits, turnaround={mg['turnaround'].get('time_sec')}", flush=True)


def build_time_index(route, run_id):
    idx = []
    wdir = WINDOW_ROOT / route / "observations" / run_id
    for p in sorted(wdir.glob("w*.json"), key=lambda p: int(p.stem[1:])) if wdir.exists() else []:
        w = json.load(open(p)); base = w["window_start_s"]; o = w["observation"]
        for lm in o.get("landmarks", []):
            idx.append({"kind": "landmark", "name": lm["name"], "side": lm.get("side"), "detail": lm.get("detail_visible"),
                        "start_sec": round(base + gen_mmss(lm["first_seen"])), "end_sec": round(base + gen_mmss(lm["last_seen"]))})
        for ev in o.get("events", []):
            idx.append({"kind": "event", "name": ev["description"], "sec": round(base + gen_mmss(ev["at"]))})
        for t in o.get("turns", []):
            idx.append({"kind": "turn", "name": t["kind"], "sec": round(base + gen_mmss(t["at"]))})
    return idx


def validate_global(d, dur):
    ph = d.get("route_phases")
    if not isinstance(ph, list) or not ph:
        raise ValueError("route_phases required")
    for p in ph:
        if not (0 <= float(p["start_sec"]) < float(p["end_sec"]) <= dur * 1.05):
            raise ValueError(f"bad phase span {p.get('start_sec')}-{p.get('end_sec')}")
    for k in ("turns", "revisited_landmarks", "environment_stages", "dynamic_events"):
        if not isinstance(d.get(k), list):
            raise ValueError(f"{k} must be a list")
    if not isinstance(d.get("start_end_relation"), dict):
        raise ValueError("start_end_relation required")


def _common_checks(q, opts, ans):
    if not isinstance(opts, dict) or sorted(opts) != [chr(65 + i) for i in range(len(opts))]:
        raise ValueError("options must be A.. consecutive")
    if not isinstance(ans, list) or not ans or not set(ans) <= set(opts):
        raise ValueError("answer must be a non-empty subset of options")
    want = TARGET_MULTI if len(ans) > 1 else TARGET_SINGLE
    if len(opts) != want:
        raise ValueError(f"expected {want} options, got {len(opts)}")
    low = (q["question"] + " " + " ".join(opts.values())).lower()
    hit = [w for w in LEAK if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low)]
    if hit:
        raise ValueError(f"leak: {hit[0]!r}")
    words = {k: len(v.split()) for k, v in opts.items()}
    longest = max(words.values())
    if all(words[k] == longest for k in ans) and sum(1 for w in words.values() if w == longest) == len(ans):
        raise ValueError("the correct option(s) are the longest — length cue")
    stem_nums = set(re.findall(r"(?<!\d)(\d{2,4})(?!\d)", q["question"]))
    if stem_nums & set(n for k in ans for n in re.findall(r"(?<!\d)(\d{2,4})(?!\d)", opts[k])):
        raise ValueError("stem leaks the answer timestamp")


def normalise_options(q):
    if isinstance(q.get("options"), list) and all(isinstance(o, str) for o in q["options"]):
        q["options"] = {chr(65 + i): o for i, o in enumerate(q["options"])}


OUT_OF_SCENE = ("beach", "highway", "motorway", "forest", "woods", "indoor", "track", "boardwalk", "figure-eight",
                "figure eight", "bridge", "tunnel", "stadium", "mall", "subway", "desert", "mountain", "snow", "shop",
                "supermarket", "playground", "cemetery", "church", "castle", "harbor", "harbour", "airport")


def novel_scene_words(text, scene_vocab):
    low = text.lower()
    return [w for w in OUT_OF_SCENE if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low) and w not in scene_vocab]


def whole_question_check(dur, scene_vocab, turnaround=None):
    def check(q):
        if q.get("question_type") not in WHOLE_TYPES:
            raise ValueError("bad question_type")
        _common_checks(q, q.get("options"), q.get("answer"))
        for k, v in q["options"].items():
            if k not in q["answer"]:
                bad = novel_scene_words(v, scene_vocab)
                if bad:
                    raise ValueError(f"distractor introduces out-of-scene content {bad[:2]}")
        norm = [re.sub(r"\W+", " ", str(v).lower()).strip() for v in q["options"].values()]
        if len(set(norm)) != len(norm):
            raise ValueError("options are not textually distinct")
        sp = q.get("evidence_spans")
        if not isinstance(sp, list) or not sp:
            raise ValueError("evidence_spans required")
        for s in sp:
            if not (isinstance(s, list) and len(s) == 2 and 0 <= s[0] < s[1] <= dur * 1.05 and s[1] - s[0] <= 120):
                raise ValueError(f"evidence span {s} must be 0..{int(dur)} and ≤120 s long")
        if turnaround:
            legs = {"out" if (s[0] + s[1]) / 2 < turnaround else "back" for s in sp}
            if q["question_type"] in ("landmark-order", "multi-step-turns") and len(legs) > 1:
                raise ValueError(f"out-and-back route: order/turn question mixes outbound and return legs (turnaround {int(turnaround)} s) — restrict to one leg")
            if q["question_type"] == "landmark-revisit" and (len(sp) < 2 or len(legs) < 2):
                raise ValueError("landmark-revisit needs evidence on both the outbound and the return leg")
    return check


def whole_batch_check(n):
    def check(qs):
        if len(qs) < max(3, n // 2):
            raise ValueError(f"only {len(qs)} valid questions after top-up")
        per_type = {}
        for q in qs:
            per_type[q["question_type"]] = per_type.get(q["question_type"], 0) + 1
        # batch-level preferences are soft here: drop extras of an over-represented type
        while any(c > 2 for c in per_type.values()):
            t = max(per_type, key=per_type.get)
            idx = max(i for i, q in enumerate(qs) if q["question_type"] == t); qs.pop(idx); per_type[t] -= 1
    return check


def cross_question_check(vids, durs, scene_vocab):
    def check(q):
        if q.get("question_type") not in CROSS_TYPES:
            raise ValueError("bad question_type")
        _common_checks(q, q.get("options"), q.get("answer"))
        for k, v in q["options"].items():
            if k not in q["answer"]:
                bad = novel_scene_words(v, scene_vocab)
                if bad:
                    raise ValueError(f"distractor introduces out-of-scene content {bad[:2]}")
        req = q.get("required_video_ids")
        if not isinstance(req, list) or len(req) < 2 or not set(req) <= set(vids):
            raise ValueError("required_video_ids must name ≥2 videos")
        ev = q.get("evidence")
        if not isinstance(ev, dict) or any(v not in ev or not ev[v] for v in req):
            raise ValueError("evidence missing for a required video")
        for v, spans in ev.items():
            for s in spans:
                if not (isinstance(s, list) and len(s) == 2 and 0 <= s[0] < s[1] <= durs[v] * 1.05 and s[1] - s[0] <= 120):
                    raise ValueError(f"evidence span {s} for {v} must be within the video and ≤120 s")
        if q["question_type"] == "landmark-order-discrimination" and any(len(ev[v]) < 2 for v in req):
            raise ValueError("landmark-order-discrimination needs one window per landmark in every video")

    return check


cross_batch_check = whole_batch_check


def generate_topup(api, gen, prompt_fn, per_question_check, batch_check, n, max_tokens=32768, rounds=6):
    """Ask for n questions; keep every question that passes on its own; re-ask only for the
    shortfall, telling the model exactly why each rejected question failed. All-or-nothing
    validation of a 6-question batch needed >8 rounds in pilot 4 (each fix broke another rule).
    `batch_check(kept)` enforces batch-level constraints (multi-answer count, ≤2 per type) at the end."""
    kept, feedback = [], ""
    for rnd in range(rounds):
        need = n - len(kept)
        if need <= 0:
            break
        try:
            raw = api.generate(gen.STRUCTURE_MODEL, [{"text": prompt_fn(need, kept, feedback)}], max_tokens, json_mode=True)
            data = gen.clean_json(raw)
        except Exception as exc:
            feedback = f"Your previous response was not valid JSON ({type(exc).__name__})."; continue
        qs = data.get("questions") if isinstance(data, dict) else None
        if not isinstance(qs, list):
            feedback = "Return {\"questions\": [...]}."; continue
        rejected = []
        for q in qs:
            try:
                normalise_options(q); per_question_check(q); kept.append(q)
            except Exception as exc:
                rejected.append(f"- \"{str(q.get('question', ''))[:70]}\" → {exc}")
            if len(kept) >= n:
                break
        feedback = ("These questions were REJECTED; write different ones that avoid these problems:\n" + "\n".join(rejected)) if rejected else ""
        print(f"    top-up round {rnd + 1}: kept {len(kept)}/{n}, rejected {len(rejected)}", flush=True)
    kept = kept[:n]
    batch_check(kept)
    return kept


def validate_blind(d):
    if not isinstance(d.get("answer"), list) or not d["answer"]:
        raise ValueError("answer must be a non-empty list")


def hires_clip(run, s, e, out, tag):
    """480p / crf 30 clip around an evidence span — sharper than the 360p global input."""
    s2, e2 = max(0, s - 5), min(run["duration_sec"], e + 5)
    clip = out / "recheck" / f"{tag}_{run['run_id']}_{int(s2)}_{int(e2)}.480p.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    if not clip.exists():
        ffmpeg("-ss", s2, "-i", run["file"], "-t", e2 - s2, "-vf", "scale=-2:480", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "30", "-an", "-movflags", "+faststart", clip)
    return clip, s2, e2


def run_gates(q, clips, api, gen, letters_text):
    """clips: list of (label, path). Mutates q: adds gates, shuffles options."""
    gen.shuffle_options([q]); q["option_order"] = "shuffled"
    q["gates"] = {"structure": "pass", "leak_scan": "pass"}
    arity = "Exactly one option is correct." if len(q["answer"]) == 1 else "More than one option is correct."
    try:
        hits, guesses = 0, []
        for t in range(3):
            bg = gen.generate_valid_json(api, BLIND_PROMPT.format(question=q["question"],
                 options=json.dumps(q["options"], ensure_ascii=False), arity=arity + f" (attempt {t + 1})"), validate_blind, 4096)
            guesses.append(sorted(bg["answer"])); hits += sorted(bg["answer"]) == sorted(q["answer"])
        q["gates"]["blind_guess"] = {"guesses": guesses, "hits_of_3": hits, "matches_gold": hits >= 2}
    except Exception as exc:
        q["gates"]["blind_guess"] = {"error": type(exc).__name__}
    if q["gates"]["blind_guess"].get("matches_gold"):
        # Guessable without video: it will be dropped regardless, so the (expensive) video
        # re-check buys nothing. ~20% of recheck calls at pilot rates.
        q["gates"]["visual_recheck"] = {"verdict": "skipped", "notes": "blind-guessable; not re-checked"}
        return
    parts = [{"text": RECHECK_PROMPT.format(question=q["question"], options=json.dumps(q["options"], ensure_ascii=False), answer=q["answer"])}]
    total, dropped = 0, []
    for label, path in clips:
        size = path.stat().st_size
        if total + size > 20 * 1024 * 1024:
            dropped.append(label); continue
        total += size
        parts.append({"text": f"[clip: {label}]"})
        parts.append({"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(path.read_bytes()).decode("ascii")}})
    if dropped:
        q["recheck_dropped_clips"] = dropped
    try:
        q["gates"]["visual_recheck"] = gen.clean_json(api.generate(gen.CAPTION_MODEL, parts, 8192, json_mode=True))
    except Exception as exc:
        q["gates"]["visual_recheck"] = {"verdict": "error", "notes": type(exc).__name__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--n-whole", type=int, default=6)
    ap.add_argument("--n-cross", type=int, default=8)
    ap.add_argument("--batch", default="", help="suffix for a top-up batch, e.g. v2. Writes "
                    "<run>.whole_video.<batch>.json alongside the existing set instead of skipping "
                    "it, and feeds every earlier question in as already-accepted so the model does "
                    "not reuse the same landmarks and events.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-gates", action="store_true")
    a = ap.parse_args()

    gen = load_generator()
    scenes = json.load(open(SCENES))
    runs = sorted([r for r in scenes["runs"] if r["route_id"] == a.route], key=lambda r: r["run_id"])
    if not runs:
        raise SystemExit(f"no runs for {a.route}")
    out = OUT_ROOT / a.route
    letters = {chr(65 + i): r for i, r in enumerate(runs)}
    print(f"{a.route} (scene {runs[0]['scene_id']}): {len(runs)} runs")
    for L, r in letters.items():
        print(f"  {L} = {r['run_id']}  {r['duration_sec']:.0f}s  {r['shape']}  {r['direction']}")
    if a.dry_run:
        n_cross = 1 if len(runs) >= 2 else 0
        print(f"video calls: {len(runs)} global + ~{len(runs) * a.n_whole + a.n_cross} rechecks | text calls: ~{len(runs) + n_cross + 3 * (len(runs) * a.n_whole + a.n_cross)}")
        return 0

    # The mounted client certificate is what Floodgate authenticates; the project token is
    # accounting metadata and the endpoint answers without it. Pass one when it is set.
    api = gen.Floodgate(os.environ.get("FLOODGATE_PROJECT_TOKEN", "")); api.session = PacedSession(api.session, RateLimiter(QUOTA_PER_SECOND))
    (out / "annotations").mkdir(parents=True, exist_ok=True); (out / "questions").mkdir(exist_ok=True)

    # ---- 1+2: global pass + time index ----
    annotations = {}
    for L, r in letters.items():
        ann_path = out / "annotations" / f"{r['run_id']}.json"
        if ann_path.exists():
            ann = json.load(open(ann_path)); changed = False
            # Retro-fit: annotations made before the merge step have empty revisits on split inputs.
            if ann.get("global_input_split") and "source" not in (ann.get("start_end_relation") or {}):
                try:
                    merge_halves(api, gen, r, ann, int(ann["global_input_split"][0][1]), L); changed = True
                except Exception as exc:
                    print(f"  merge (retro) failed {L}: {type(exc).__name__}", flush=True)
            # Refresh the internal time index if window observations appeared since.
            idx_now = build_time_index(a.route, r["run_id"])
            if len(idx_now) > len(ann.get("time_index", {}).get("entries", [])):
                ann["time_index"] = {"source": "3-minute window observations (internal index, not samples)", "entries": idx_now}; changed = True
                print(f"  index refresh {L}: {len(idx_now)} entries", flush=True)
            if changed:
                gen.write_json(ann_path, ann)
            annotations[L] = ann; continue
        pieces = global_inputs(r, out)
        merged = {"route_phases": [], "turns": [], "revisited_landmarks": [], "environment_stages": [],
                  "dynamic_events": [], "uncertainties": [], "start_end_relation": None}
        failed = False
        for k, (gi, p0, p1) in enumerate(pieces):
            enc = base64.b64encode(gi.read_bytes()).decode("ascii")
            part_desc = "the COMPLETE" if len(pieces) == 1 else f"part {k + 1} of {len(pieces)}"
            piece_ann = None
            for attempt in range(4):
                try:
                    raw = api.generate(gen.CAPTION_MODEL, [{"text": GLOBAL_PROMPT.format(vid=L, dur=int(r["duration_sec"]),
                                                            part_desc=part_desc, p0=int(p0), p1=int(p1))},
                                                           {"inlineData": {"mimeType": "video/mp4", "data": enc}}], 16384, json_mode=True)
                    piece_ann = gen.clean_json(raw); validate_global(piece_ann, r["duration_sec"]); break
                except Exception as exc:
                    print(f"  global retry {attempt + 1}/4 {L} part{k + 1}: {type(exc).__name__}: {str(exc)[:90]}", flush=True); piece_ann = None
            if piece_ann is None:
                failed = True; break
            for key in ("route_phases", "turns", "revisited_landmarks", "environment_stages", "dynamic_events", "uncertainties"):
                merged[key].extend(piece_ann.get(key) or [])
            # start/end relation only makes sense on the last part (it sees the end) — but it also needs
            # the start; for split videos record the last part's judgement and flag it.
            merged["start_end_relation"] = piece_ann.get("start_end_relation") or merged["start_end_relation"]
        if failed:
            print(f"  !! global pass failed for {L}; skipping run", flush=True); continue
        if len(pieces) > 1:
            merged["global_input_split"] = [[int(p0), int(p1)] for _, p0, p1 in pieces]
            try:
                merge_halves(api, gen, r, merged, int(pieces[0][2]), L)
            except Exception as exc:
                merged["start_end_relation"] = dict(merged["start_end_relation"] or {}, note="judged from part 2 only; merge failed")
                print(f"  merge failed {L}: {type(exc).__name__}", flush=True)
        glob_ann = merged
        # time index from existing window observations (internal only)
        idx = []
        wdir = WINDOW_ROOT / a.route / "observations" / r["run_id"]
        for p in sorted(wdir.glob("w*.json"), key=lambda p: int(p.stem[1:])) if wdir.exists() else []:
            w = json.load(open(p)); base = w["window_start_s"]; o = w["observation"]
            for lm in o.get("landmarks", []):
                idx.append({"kind": "landmark", "name": lm["name"], "side": lm.get("side"), "detail": lm.get("detail_visible"),
                            "start_sec": round(base + gen_mmss(lm["first_seen"])), "end_sec": round(base + gen_mmss(lm["last_seen"]))})
            for ev in o.get("events", []):
                idx.append({"kind": "event", "name": ev["description"], "sec": round(base + gen_mmss(ev["at"]))})
        ann = dict(video_id=r["run_id"], duration_sec=r["duration_sec"], scene_id=r["scene_id"], route_id=r["route_id"],
                   run_meta={k: r[k] for k in ("participant", "direction", "shape", "out_and_back", "turnaround_sec", "gpx", "speed", "lighting")},
                   global_input=[str(gi.relative_to(out)) for gi, _, _ in pieces], global_model=gen.CAPTION_MODEL,
                   global_input_split=glob_ann.get("global_input_split"),
                   **{k: glob_ann.get(k) for k in ("route_phases", "turns", "start_end_relation", "revisited_landmarks",
                                                   "environment_stages", "dynamic_events", "uncertainties")},
                   time_index={"source": "3-minute window observations (internal index, not samples)", "entries": idx},
                   created_utc=gen.utc_now())
        gen.write_json(ann_path, ann); annotations[L] = ann
        print(f"  global {L}: {len(ann['route_phases'])} phases, {len(ann['turns'])} turns, returns_near_start={ann['start_end_relation'].get('returns_near_start')}, index {len(idx)}", flush=True)

    def public(ann):   # what the question writer sees: no run_meta (speed/lighting), no file names
        se = dict(ann["start_end_relation"] or {})
        shape = ann["run_meta"].get("shape")
        if shape in ("out_and_back", "loop", "one_way"):
            se["gpx_truth"] = {"out_and_back": "returns near start by retracing", "loop": "returns near start via a circuit",
                               "one_way": "does NOT return near start"}[shape]
        facts = {}
        if shape == "out_and_back":
            ta = ann["run_meta"].get("turnaround_sec")
            facts["turnaround_sec"] = ta
            facts["route_geometry"] = (f"OUT-AND-BACK: the route reverses at about {ta} s. Every landmark seen before {ta} s is passed "
                                       "again afterwards, in reverse order and on the opposite side.") if ta else \
                                      "OUT-AND-BACK: the route retraces itself; landmarks are passed twice with sides swapped."
        elif shape == "loop":
            facts["route_geometry"] = "LOOP: returns near the start via a circuit; no retracing."
        elif shape == "one_way":
            facts["route_geometry"] = "ONE-WAY: does not return; every landmark is passed once."
        return {k: ann[k] for k in ("duration_sec", "route_phases", "turns", "revisited_landmarks",
                                    "environment_stages", "dynamic_events")} | {"start_end_relation": se, "facts": facts,
                                                                                 "time_index": ann["time_index"]["entries"]}

    def scene_vocab_of(anns):
        return frozenset(w for x in anns for w in re.findall(r"[a-z][a-z\-]{2,}", json.dumps(public(x)).lower()))

    # ---- 3: whole-video questions ----
    suffix = f".{a.batch}" if a.batch else ""

    def prior_questions(*paths):
        """Questions already written for this unit, whatever batch produced them."""
        out_qs = []
        for path in paths:
            if path.exists():
                out_qs += json.load(open(path)).get("questions") or []
        return out_qs

    for L, ann in annotations.items():
        r = letters[L]
        qpath = out / "questions" / f"{r['run_id']}.whole_video{suffix}.json"
        if qpath.exists():
            continue
        earlier = prior_questions(*(out / "questions").glob(f"{r['run_id']}.whole_video*.json"))
        siblings = {M: {"route_phases": [p.get("description") for p in x["route_phases"]],
                        "landmarks": sorted({lm for p in x["route_phases"] for lm in p.get("landmarks", [])})[:20]}
                    for M, x in annotations.items() if M != L}
        try:
            vocab = scene_vocab_of(list(annotations.values()))
            def whole_prompt(need, kept, feedback, _ann=ann, _r=r, _L=L, _sib=siblings):
                p = WHOLE_Q_PROMPT.format(n=need, dur=int(_r["duration_sec"]), vid=_L,
                                          annotation=json.dumps(public(_ann), ensure_ascii=False), n_multi=max(1, need // 3),
                                          siblings=json.dumps(_sib, ensure_ascii=False) or "none",
                                          single=TARGET_SINGLE, multi=TARGET_MULTI)
                seen = earlier + kept
                if seen:
                    p += "\n\nALREADY ACCEPTED (do not repeat their landmarks/events): " + json.dumps([k["question"] for k in seen], ensure_ascii=False)
                return p + ("\n\n" + feedback if feedback else "")
            qs = generate_topup(api, gen, whole_prompt,
                                whole_question_check(r["duration_sec"], vocab,
                                                     turnaround=r.get("turnaround_sec") if r.get("shape") == "out_and_back" else None),
                                whole_batch_check(a.n_whole), a.n_whole)
        except Exception as exc:
            print(f"  !! whole-video questions failed for {L}: {type(exc).__name__}", flush=True); continue
        for i, q in enumerate(qs, 1):
            q["video_id"] = r["run_id"]; q["scene_id"] = r["scene_id"]; q["route_id"] = r["route_id"]
            if not a.skip_gates:
                clips = []
                for (s, e) in q["evidence_spans"][:4]:
                    c, s2, e2 = hires_clip(r, s, e, out, f"w{L}{i}"); clips.append((f"{int(s2)}-{int(e2)}s", c))
                run_gates(q, clips, api, gen, L)
                v = q["gates"]["visual_recheck"]; print(f"  {L} whole q{i} [{q['question_type']}] recheck={v.get('verdict') if isinstance(v, dict) else v} blind={q['gates']['blind_guess'].get('matches_gold')}", flush=True)
        gen.write_json(qpath, {"video_id": r["run_id"], "unit": "whole_video", "questions": qs})

    # ---- 4: cross-video questions ----
    if len(annotations) >= 2:
        cpath = out / "questions" / f"{a.route}.cross_video{suffix}.json"
        if not cpath.exists():
            earlier_cross = prior_questions(*(out / "questions").glob(f"{a.route}.cross_video*.json"))
            vids = ", ".join(annotations)
            durs = {L: letters[L]["duration_sec"] for L in annotations}
            try:
                vocab = scene_vocab_of(list(annotations.values()))
                def cross_prompt(need, kept, feedback):
                    p = CROSS_Q_PROMPT.format(n=need, vids=vids,
                                              annotations=json.dumps({L: public(x) for L, x in annotations.items()}, ensure_ascii=False),
                                              n_multi=max(1, need // 3), single=TARGET_SINGLE, multi=TARGET_MULTI)
                    seen = earlier_cross + kept
                    if seen:
                        p += "\n\nALREADY ACCEPTED (do not repeat their landmarks/events): " + json.dumps([k["question"] for k in seen], ensure_ascii=False)
                    return p + ("\n\n" + feedback if feedback else "")
                qs = generate_topup(api, gen, cross_prompt, cross_question_check(list(annotations), durs, vocab), cross_batch_check(a.n_cross), a.n_cross)
            except Exception as exc:
                print(f"  !! cross-video questions failed: {type(exc).__name__}", flush=True); qs = []
            for i, q in enumerate(qs, 1):
                q["scene_id"] = runs[0]["scene_id"]; q["route_id"] = a.route
                q["video_ids"] = {L: letters[L]["run_id"] for L in q["required_video_ids"]}
                if not a.skip_gates:
                    clips = []
                    named = [L for L in q["required_video_ids"] if any(f"video {L}".lower() in q["options"][k].lower() for k in q["answer"])]
                    for L in named + [L for L in q["required_video_ids"] if L not in named]:
                        for (s, e) in q["evidence"][L][:3]:
                            c, s2, e2 = hires_clip(letters[L], s, e, out, f"x{i}"); clips.append((f"video {L}, {int(s2)}-{int(e2)}s", c))
                    run_gates(q, clips, api, gen, vids)
                    v = q["gates"]["visual_recheck"]; print(f"  cross q{i} [{q['question_type']}] recheck={v.get('verdict') if isinstance(v, dict) else v} blind={q['gates']['blind_guess'].get('matches_gold')}", flush=True)
            gen.write_json(cpath, {"route_id": a.route, "unit": "cross_video", "anonymous_map": {L: letters[L]["run_id"] for L in annotations}, "questions": qs})

    # ---- summary ----
    tot = ok = blind = 0
    for p in (out / "questions").glob("*.json"):
        for q in json.load(open(p))["questions"]:
            tot += 1; g = q.get("gates", {}); v = g.get("visual_recheck", {})
            ok += isinstance(v, dict) and v.get("verdict") == "supported"; blind += bool(g.get("blind_guess", {}).get("matches_gold"))
    print(f"\n{a.route}: {tot} questions, {ok} supported by high-res recheck, {blind} blind-guessable → {out}")
    return 0


def gen_mmss(s):
    parts = [float(x) for x in str(s).strip().split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0]


if __name__ == "__main__":
    raise SystemExit(main())
