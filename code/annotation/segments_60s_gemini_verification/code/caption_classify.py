# -*- coding: utf-8 -*-
"""基于 caption 的题型分类。

依据 segments60s_kit/README.md: "captions/segments_60s.jsonl ... 题目就是照这些写的"。
所以题目的类型 = 它派生自 caption 的哪个字段, 而不是靠模型重看视频去猜。

caption 字段 -> 类别:
  overall_environment      -> Scene and Place QA
  objects_and_attributes   -> Object and Attribute QA
  spatial_relations        -> Ego-relative Spatial Relation
  trajectory_and_turnings  -> Trajectory-grounded QA / Ego-motion Recognition
  actions_and_events       -> Ego-motion Recognition / Actor Action and Motion (按主语)

Text and OCR / Temporal Grounding / Event Sequencing 不是 caption 字段, 而是提问形式,
故先于字段归因判定。
"""
import json, os, re, sys, collections

KIT = '/mnt/data/cvhci_video_understanding/segments60s_kit'

STOP = set(('a an the of on in at to and or with is are was were be been it its this that these those '
            'side left right along near during video camera path which following correct describe describes '
            'true visual details observed seen located from for by as their there also both while when what '
            'where how towards toward into over under about across then than they them his her one two').split())


def toks(s):
    return {w for w in re.findall(r'[a-z]+', (s or '').lower()) if w not in STOP and len(w) > 3}


def flat(v):
    """caption 字段可能是 str / list / dict, 递归拍平成文本"""
    if v is None:
        return ''
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ' '.join(flat(x) for x in v)
    if isinstance(v, dict):
        return ' '.join(flat(x) for x in v.values())
    return str(v)


# --- 提问形式判据 (先于字段归因) ---
OCR = re.compile(r'\b(text|word|words|number|numbers|letter|letters|written|reads?|says|displayed|'
                 r'inscription|lettering)\b', re.I)
QUOTED = re.compile(r"['‘’\"]\s*[\w.\- ]{1,30}\s*['‘’\"]")
TIME = re.compile(r'\b\d{1,2}:\d{2}\b|\bat \d{1,2} ?(?:seconds?|s)\b')
TIME_Q = re.compile(r'at (?:approximately )?what time|at what point|when does|how (?:long|many seconds)', re.I)
SEQ_Q = re.compile(r'\b(sequence|order|chronological|followed by)\b', re.I)
REGION = re.compile(r'\b(left|right)[- ]?(?:hand )?side\b|directly ahead|directly behind|'
                    r'\bto the (?:left|right)\b|\b(?:left|right) of\b|relative to|'
                    r'spatial (?:relation|arrangement|layout)', re.I)
WHOLE = re.compile(r'throughout|overall|from (?:the )?start to (?:the )?end|trajectory|movement pattern|'
                   r'what (?:sequence|series) of|over the course of', re.I)
EGO_SUBJ = re.compile(r'\b(?:the )?camera(?:\'s|’s)?\b|camera wearer|camera operator|camera person|'
                      r'\bthe runner\b|\bthe person\b|\bthey\b|\byou\b', re.I)
EGO_OBJ = re.compile(r'\b(?:away from|towards?|past|behind|in front of|at) the camera\b', re.I)
ACTOR = re.compile(r'\b(cyclist|pedestrian|jogger|man|woman|child|dog|car|van|truck|tram|bus|scooter|'
                   r'bicycle|vehicle|bird|people|figures)\b', re.I)


def timestamp_necessary(q):
    if not any(TIME.search(v) for v in q['options'].values()):
        return False
    bare = {k: TIME.sub('', v).strip() for k, v in q['options'].items()}
    for k in q['answer']:
        for k2, v2 in bare.items():
            if k2 != k and v2 == bare[k]:
                return True
    return bool(TIME_Q.search(q['question']))


def permutation_like(q):
    ts = [toks(v) for v in q['options'].values()]
    big = [t for t in ts if len(t) >= 4]
    if len(big) < 4:
        return False
    sims = []
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            u = len(big[i] | big[j])
            sims.append(len(big[i] & big[j]) / u if u else 0)
    return sorted(sims)[len(sims) // 2] >= 0.55


def field_scores(q, cap):
    """正确项与 caption 各字段的词汇重合度"""
    ans = set()
    for k in q['answer']:
        ans |= toks(q['options'][k])
    if not ans:
        return {}
    out = {}
    for name, key in (('env', 'overall_environment'), ('objects', 'objects_and_attributes'),
                      ('spatial', 'spatial_relations'), ('traj', 'trajectory_and_turnings'),
                      ('actions', 'actions_and_events')):
        out[name] = len(ans & toks(flat(cap.get(key)))) / len(ans)
    return out


def distractors_in_caption(q, cap):
    """干扰项里的物体是否出现在 caption 中 -> 决定 Spatial Relation vs Object QA"""
    full = toks(flat(cap))
    dis = [k for k in q['options'] if k not in q['answer']]
    hit = n = 0
    for k in dis:
        t = toks(q['options'][k])
        if len(t) < 2:
            continue
        n += 1
        if len(t & full) / len(t) >= 0.5:
            hit += 1
    return hit / n if n >= 2 else None


def classify(q, cap):
    # 1. 提问形式
    if OCR.search(q['question']) and QUOTED.search(' '.join(q['options'].values())):
        return 'Text and OCR', 'form:ocr'
    if timestamp_necessary(q):
        return 'Temporal Grounding', 'form:timestamp-necessary'
    if permutation_like(q) and SEQ_Q.search(q['question']):
        return 'Event Sequencing', 'form:permutation'

    s = field_scores(q, cap)
    if not s:
        return 'Object and Attribute QA', 'no-tokens'
    best = max(s, key=s.get)

    # 2. 自身运动 / 外部对象
    # 字段归因优先于主语检测: 轨迹类选项常省略主语("Starts by moving straight, turns right"),
    # 只靠主语正则会漏判, 但它命中 trajectory_and_turnings 这件事本身就是证据。
    ans_txt = ' '.join(q['options'][k] for k in q['answer'])
    ego = bool(EGO_SUBJ.search(EGO_OBJ.sub(' ', ans_txt)))
    actor = bool(ACTOR.search(ans_txt)) and not ego

    if best == 'traj' or s['traj'] >= 0.30:
        if actor and s['actions'] > s['traj']:
            return 'Actor Action and Motion', f'field:{best}/actor'
        return ('Trajectory-grounded QA' if (WHOLE.search(q['question']) or s['traj'] >= s['actions'])
                else 'Ego-motion Recognition'), f'field:traj'
    if best == 'actions':
        if actor:
            return 'Actor Action and Motion', 'field:actions/actor'
        if ego or s['actions'] >= 0.30:
            return ('Trajectory-grounded QA' if WHOLE.search(q['question'])
                    else 'Ego-motion Recognition'), 'field:actions/ego'

    # 3. 区域限定题: 干扰项在不在 caption 里
    if REGION.search(q['question']):
        dp = distractors_in_caption(q, cap)
        if dp is not None and dp >= 0.5:
            return 'Ego-relative Spatial Relation', f'region/distractors-present={dp:.2f}'
        if dp is not None and dp <= 0.25:
            return 'Object and Attribute QA', f'region/distractors-absent={dp:.2f}'

    if best == 'spatial' and s['spatial'] > s['objects']:
        return 'Ego-relative Spatial Relation', 'field:spatial'
    if best == 'env' and s['env'] > s['objects']:
        return 'Scene and Place QA', 'field:env'
    return 'Object and Attribute QA', f'field:{best}'


def load_captions():
    caps = {}
    for l in open(f'{KIT}/captions/segments_60s.jsonl'):
        c = json.loads(l)
        caps[(c['video'], tuple(c['span_sec']))] = c['caption']
    return caps


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'segments_60s_typed.jsonl'
    out = sys.argv[2] if len(sys.argv) > 2 else 'caption_labels.jsonl'
    caps = load_captions()
    rows = [json.loads(l) for l in open(src)]
    miss = 0
    with open(out, 'w') as f:
        for r in rows:
            cap = caps.get((r['video'], tuple(r['span_sec'])))
            if cap is None:
                miss += 1
                lab, why = 'Object and Attribute QA', 'no-caption'
            else:
                lab, why = classify(r, cap)
            f.write(json.dumps({'id': r['id'], 'label': lab, 'why': why}, ensure_ascii=False) + '\n')
    print(f'{len(rows)} 题 -> {out}  (无 caption {miss})')
    print(collections.Counter(json.loads(l)['label'] for l in open(out)).most_common())
