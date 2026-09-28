> This workflow has **no blind-guess step**: watch first, then check every option. Ignore the paragraphs about "blind / Step 1"; every judging rule below applies unchanged.

# How to Watch and Judge — Reviewer's Manual

The README covers the process (three steps, tool usage). This file covers the craft:
once a clip is in front of you, what your eyes should look for, what your hand should
write down, and how to settle each option.

---

## 0. One overriding principle

> **Check every option against the footage. Never judge on overall impression.**

Wrong way: watch the clip, then pick "the three that feel right".
Right way: **go through all 8 options one at a time.** For each one, find the frame that
supports it or the frame that rules it out. If you find neither, mark it undecidable.

Why: these distractors are deliberate near-misses — they take something that really is on
screen and change one or two details (a side swapped, an order reversed, a colour wrong).
On an impression pass every near-miss looks familiar and you miss the altered detail.
**Only option-by-option checking catches it.**

---

## 1. Two passes

### Pass one — watch straight through, build a timeline

Set the options aside. Play the clip end to end and **write a timeline as you go**:

```
0:05  start, asphalt road, streetlights on
0:18  right turn, onto cobblestone
0:31  building marked 11 on the left
0:44  oncoming cyclist with headlight
0:58  passing bicycle racks
1:12  through two striped bollards
1:27  concrete benches on the right
1:40  red-and-white construction barriers begin
```

What to note: **turns (direction and roughly how sharp), surface changes, anything with a
number, text or distinctive colour, moving people and vehicles, prominent buildings.**
You do not need everything — you need the things a question might be built on.

Timestamps need no precision; ±3 seconds is plenty. What you want is **order and rough
position**, not stopwatch accuracy.

For multi-clip questions (the stem names labels like CLIP_B, CLIP_F): **keep a separate
timeline per clip**, with the label at the top. Note that **label order does not imply
time order** — CLIP_A is not necessarily earlier than CLIP_B.

### Pass two — check each option at fixed points

Now open the options and take them one at a time:

1. **Break out what it claims.** For example option B: "passing between two striped
   bollards onto a path with concrete benches occurs earlier than the red-and-white
   construction barriers" breaks into three claims — ① there are striped bollards
   ② benches follow them ③ both come before the barriers.
2. **Check against your timeline.** Bollards 1:12, benches 1:27, barriers 1:40 — all
   three hold → supported.
3. **Unsure? Scrub back.** Jump to the matching point in your timeline and watch just
   those few seconds. This is what the timeline is for: pass two needs no re-watch, only
   spot checks.

Mark each option on paper: **✓ supported · ✗ ruled out (note which detail is wrong) ·
? undecidable**

Finish all 8 before concluding. **✓ should come to exactly 3** — more or fewer is itself
a finding (either the annotation is wrong or the question is ambiguous), and reporting it
is the point.

---

## 2. What to look for, per question type

| Type | Note in pass one | Check in pass two |
|---|---|---|
| `event-order` | timestamp of each salient event | every "X before Y" claim against the timeline |
| `turn-pattern` | each turn: direction, sharpness, junction features | whether the stated direction/angle/sequence is reversed |
| `landmark-order` | when each landmark appears **and which side** | order **and side** (side is the detail most often altered) |
| `landmark-revisit` | when a thing reappears, and the direction | whether it really is the same object (colour, text, damage) |
| `environment-*` | where surface, vegetation or building density changes | the direction of change ("asphalt→gravel" or the reverse) |
| `spatial-consistency` | key objects left / right / ahead | **on an out-and-back, right on the way out is left on the way back** |
| `same-place-different-recording` | a landmark list per clip | landmark by landmark: the same tree? the same sign? |
| `route-identity` | surface and turn sequence per clip | only a matching turn sequence proves the same route; one similar landmark does not |

**For cross-recording questions** (comparing two separate takes), also note:
- Lighting can differ completely (one by day, one at night) — do not call it a different
  place just because it "looks unlike"
- To judge **same place**, rely on what does not change: building shape, the path's
  geometry, fixed fixtures
- To judge **different**, rely on **structural** differences — a different turn sequence,
  a different surface — not on incidental pedestrians and vehicles

---

## 3. Verdict rules (three easy ones to get wrong)

### ① The option names something the footage does not contain

**It depends, and this is the most commonly misjudged case:**

- The clip **does cover** the place or moment the option is about, and the thing is not
  there → **✗ ruled out**. Example: the option says "a dark grey station wagon parked by
  the benches"; the clip clearly shows the bench area and there is no vehicle → ruled out.
- The clip **never covers** that place or moment → **? undecidable**. Example: the option
  talks about "after crossing the bridge" and the clip ends before the bridge → undecidable.

The difference is whether **the footage gave you the chance to rule it out.**

### ② The option is mostly right, wrong in one detail

**→ ✗ ruled out.** One wrong detail is wrong. Near-miss distractors look exactly like
this; "benches on the right" (they are on the left) does not get a pass because there
really are benches.

### ③ Your count does not match the annotation's

When you count 4 ✓ or only 2 ✓, **do not force it to three.** Report honestly:
- give your three most confident as the "final answer"
- pick verdict `ambiguous` (more than three defensible) or `wrong_answer`
- **write in the notes which option caused it** — that is the most valuable feedback

---

## 4. A full worked example (real question)

**Type: event-order · 2 clips (CLIP_C, CLIP_F, two stretches of one night run)**

> Q: Which of the following correctly describe the chronological sequence of events along
> the route? Select three.
>
> A. The oncoming cyclist with a bright headlight is passed before reaching the building
>    marked 11
> B. Passing between two striped bollards onto a path with concrete benches occurs earlier
>    than running past the red-and-white construction barriers
> C. Running alongside the barriers occurs earlier than passing between the bollards
>    ← note: exactly the negation of B
> D. The runner meets the concrete benches after the straight paved stretch past the barriers
> E. The building marked 11 is passed on the cobblestone section before reaching the
>    bicycle racks
> F. The cyclist is encountered on the straight asphalt stretch with barriers, not on the
>    curved cobblestone path
> G. The building marked 11 is passed before encountering the cyclist on the cobblestone
>    ← the negation of A
> H. Running past the bicycle racks on cobblestone occurs before the straight paved path
>    bordered by lawns under warm streetlamps

**Pass one**, a timeline per clip:

```
CLIP_C: 0:03 cobblestone · 0:15 oncoming cyclist (headlight) · 0:29 building "11" · 0:41 bike racks
CLIP_F: 0:05 two striped bollards · 0:12 concrete benches (right) · ~0:30 lawn-lined straight (streetlamps) · 0:47 barriers
```

One thing is still missing: **which of CLIP_C and CLIP_F comes first?** The labels do not
say. You have to infer it from the footage — whether the scene at the end of CLIP_C
continues into the start of CLIP_F, or from continuity of environment — and if you cannot,
lean on the options decidable within a single clip.

**Pass two, option by option:**

```
A  cyclist (C 0:15) before building 11 (C 0:29)       → ✓ decidable within one clip
B  bollards (F 0:05) before barriers (F 0:47)         → ✓ decidable within one clip
C  the reverse of B                                   → ✗
D  benches (F 0:12) come before barriers (F 0:47); the option says after → ✗
E  building 11 (0:29) before bike racks (0:41)? direction reads right → wait, check again... ✓?
   But the annotation does not list E!
   → Re-watch 0:29–0:41: the racks were already at the frame edge around 0:26 → E is ✗
F  the cyclist is on the cobblestone stretch (C); the option says the asphalt/barriers stretch → ✗
G  the reverse of A                                   → ✗
H  bike racks (C 0:41, cobblestone) before the lawn-lined straight (F) → ✓ needs the
   cross-clip order, settled from the footage
```

✓ = A, B, H — exactly three, matching the annotation → verdict `correct`.

Note what happened at E: **the timeline almost misled me** — pass one missed the racks at
the frame edge around 0:26. So any option you are unsure about must be re-watched. **The
timeline is an index, not evidence.**

---

## 5. Common mistakes

| Mistake | Consequence |
|---|---|
| Picking three on overall impression instead of checking each option | every near-miss slips through |
| Watching first, then filling in a blind guess | the blind data is void; guessability reads too low |
| Treating every "not in the footage" as ruled out | conflates undecidable with ruled out; overstates quality |
| Forgetting the side swap on out-and-back stretches | every spatial question comes out reversed |
| Reading CLIP label order as time order | event-order questions come out reversed |
| Forcing the count (you see 4 correct, you pick 3) | hides the ambiguity that was worth reporting |
| Calling cross-recording clips "different places" because the lighting differs | every day/night comparison fails |

---

## 6. How much can you do in a day

Once you are fluent: roughly 2–3 minutes per single-clip question, 4–6 for multi-clip
comparisons. **About 80–100 questions in a 6-hour day.** The recommended 120-question
sample takes about a day and a half.

Fatigue measurably reduces near-miss detection — **do not review for more than 90 minutes
at a stretch.** Take a break and come back.
