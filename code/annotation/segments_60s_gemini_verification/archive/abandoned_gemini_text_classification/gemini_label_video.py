# -*- coding: utf-8 -*-
"""带视频的题型标注: 60s 片段 inline base64 送 Gemini。

相对文本标注的唯一增量能力: 模型能确认干扰项里的物体在视频中到底存不存在,
这正是 Spatial Relation 与 Object QA 的判据 (spec 5.2), 文本标注只能靠启发式近似。
"""
import base64, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from floodgate import generate

CLASSES = ["Object and Attribute QA", "Ego-relative Spatial Relation", "Trajectory-grounded QA",
           "Ego-motion Recognition", "Actor Action and Motion", "Scene and Place QA",
           "Temporal Grounding", "Event Sequencing", "Text and OCR"]

PROMPT = """You are shown a 60-second egocentric video clip and one multiple-choice question about it, with the correct option(s) marked. Assign EXACTLY ONE question-type label.

THE DECISIVE RULE: classify by what DISTINGUISHES the correct option(s) from the distractors — never by the wording of the question stem. The stems are template-generated and carry no signal. Watch the clip and ask: "what minimum fact must a viewer read off this video to pick the correct option over the distractors?"

USE THE VIDEO for this specific decision: when the question restricts to a region ("on the left side", "directly ahead"), check whether the distractor objects actually APPEAR in the clip.
  - Distractors appear in the clip but on the wrong side  -> Ego-relative Spatial Relation
  - Distractors are simply ABSENT from the clip           -> Object and Attribute QA

LABELS:
1. Text and OCR — must read text/letters/numbers printed on a sign, vehicle or building.
2. Temporal Grounding — the timestamp ITSELF is the thing being asked for. Times are relative to the clip start (00:00).
   NECESSITY TEST (apply it strictly, it is the most over-used label): mentally delete every timestamp from every option. If the correct option is STILL uniquely identifiable from its remaining content, this is NOT Temporal Grounding — label it by that remaining content instead.
   Watching the video makes timing salient; do NOT let that pull you here. A question is Temporal Grounding only if two options would become indistinguishable once their timestamps are removed.
3. Event Sequencing — the discriminator is ORDER; options are permutations of the same elements.
4. Trajectory-grounded QA — the camera-wearer's own path shape / route across the whole clip.
5. Ego-motion Recognition — one localized maneuver by the camera-wearer (stopping, one turn, zooming, looking down). "away from the camera" makes the camera an OBJECT, not the subject — that is not ego-motion.
6. Actor Action and Motion — what an EXTERNAL person/vehicle/animal does or which way it moves.
7. Ego-relative Spatial Relation — the discriminator is DIRECTION relative to the camera or path.
8. Scene and Place QA — place type, weather, time of day, lighting, overall environment.
9. Object and Attribute QA — existence, identity, colour, material, shape, count or state of objects.

PRIORITY when several fit, first match wins: 1 > 2 > 3 > 4 > 5 > 6 > 7 > 8 > 9.

Reply with JSON only:
{"label": "<exact label text>", "distractors_present": true|false|null, "why": "<max 20 words naming the discriminator>"}
Set distractors_present only for region-restricted questions: true if you saw the distractor objects in the clip."""


def ask(model, q, clipdir):
    vid, (st, en) = q['video'], q['span_sec']
    clip = None
    for f in os.listdir(clipdir):
        if f.startswith(f'{vid}__{st:.0f}_'):
            clip = os.path.join(clipdir, f)
            break
    if not clip:
        return None, None, 'no clip'
    b64 = base64.b64encode(open(clip, 'rb').read()).decode('ascii')
    opts = '\n'.join(f'  {k}) {v}' + ('   <-- CORRECT' if k in q['answer'] else '')
                     for k, v in q['options'].items())
    user = f"QUESTION: {q['question']}\nOPTIONS:\n{opts}\nCORRECT: {', '.join(q['answer'])}"
    o, diag = generate(model,
                       [{'text': user},
                        {'inlineData': {'mimeType': 'video/mp4', 'data': b64}}],
                       system=PROMPT, timeout=600)
    if not o:
        return None, None, diag
    lab = o.get('label', '')
    if lab not in CLASSES:
        lab = next((c for c in CLASSES if c.lower() in str(lab).lower()), None)
    return lab, o.get('distractors_present'), o.get('why', '')


if __name__ == '__main__':
    model, src, clipdir, out = sys.argv[1:5]
    workers = int(os.environ.get('RB_WORKERS', '6'))
    qs = [json.loads(l) for l in open(src)]
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                d = json.loads(l)
                if d.get('label'):
                    done.add(d['id'])
            except Exception:
                pass
    todo = [q for q in qs if q['id'] not in done]
    print(f'{model}: 共 {len(qs)}, 已完成 {len(done)}, 待跑 {len(todo)}', flush=True)
    import time
    t0 = time.time()
    with open(out, 'a') as f:
        for i in range(0, len(todo), 50):
            part = todo[i:i + 50]
            with ThreadPoolExecutor(workers) as ex:
                res = list(ex.map(lambda q: ask(model, q, clipdir), part))
            for q, (lab, dp, why) in zip(part, res):
                f.write(json.dumps({'id': q['id'], 'model': model, 'label': lab,
                                    'distractors_present': dp, 'why': why}, ensure_ascii=False) + '\n')
            f.flush()
            n = i + len(part)
            el = time.time() - t0
            print(f'  {model} {n}/{len(todo)}  {el:.0f}s  eta {el/max(1,n)*(len(todo)-n):.0f}s', flush=True)
    print(f'{model}: done -> {out}', flush=True)
