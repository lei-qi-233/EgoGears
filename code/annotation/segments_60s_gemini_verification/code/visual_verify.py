# -*- coding: utf-8 -*-
"""视觉核验: 对着画面逐选项对账。

依据 segments60s_kit/README.md §6 —— 这 6770 题只过了盲猜闸门, 视觉核验从未跑过;
同语料其它层里盲猜幸存题被视觉核验淘汰了 40.7%, 人工复核发现 23~29% 的标准答案
"画面里根本没有或是反的"。

必须用画面核, 不能用 caption 核: 题目就是照 caption 生成的, 拿 caption 核是循环论证,
caption 自身的错误会原样传给题目而检不出来。

逐选项输出 README §6 要求的三态:
  supported   画面里找得到支撑它的那几秒
  ruled_out   画面明确否定它 (左右反了/顺序反了/颜色不对, 一处细节错即为错)
  undecidable 画面根本没拍到该选项说的地点或时刻
"""
import base64, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from floodgate import generate

PROMPT = """You are auditing a multiple-choice question against the video it claims to describe. The question was written from a text description WITHOUT anyone looking at the footage, so the marked answer may be wrong, reversed, or about something not filmed.

TRUST THE FOOTAGE, NOT THE MARKED ANSWER.

For EVERY option, return one verdict:
  "supported"   — you can point to the seconds in this clip that support it
  "ruled_out"   — the clip positively contradicts it (one wrong detail is enough: left/right swapped, order reversed, wrong colour, wrong count)
  "undecidable" — the clip never shows the place or moment the option talks about

Be strict about left/right: judge from the camera's own point of view as it moves.
For each option also give `at` — the seconds in this clip where you looked (e.g. "12-18"), or null.

Then judge the question as a whole:
  "answer_correct"  — every marked-correct option is supported AND every unmarked option is ruled_out or undecidable
  "answer_wrong"    — some marked-correct option is ruled_out, or some unmarked option is clearly supported (making the key incomplete)
  "insufficient"    — the clip does not show enough to decide

Reply with JSON only:
{"options": {"A": {"verdict": "...", "at": "...", "note": "<=12 words"}, ...},
 "overall": "answer_correct|answer_wrong|insufficient",
 "reason": "<=25 words"}"""


def find_clip(q, clipdir):
    st = q['span_sec'][0]
    pre = f"{q['video']}__{st:.0f}_"
    for f in os.listdir(clipdir):
        if f.startswith(pre):
            return os.path.join(clipdir, f)
    return None


def verify(q, clipdir, model):
    clip = find_clip(q, clipdir)
    if not clip:
        return {'id': q['id'], 'overall': None, 'diag': 'no clip'}
    b64 = base64.b64encode(open(clip, 'rb').read()).decode('ascii')
    opts = '\n'.join(f'  {k}) {v}' + ('   [MARKED CORRECT]' if k in q['answer'] else '')
                     for k, v in q['options'].items())
    user = (f"QUESTION: {q['question']}\nOPTIONS:\n{opts}\n"
            f"MARKED CORRECT: {', '.join(q['answer'])}  (arity: {q['arity']})")
    o, diag = generate(model, [{'text': user},
                               {'inlineData': {'mimeType': 'video/mp4', 'data': b64}}],
                       system=PROMPT, timeout=600, max_tokens=12000)
    if not o:
        return {'id': q['id'], 'overall': None, 'diag': diag}
    return {'id': q['id'], 'model': model, 'report_class': q.get('report_class'),
            'overall': o.get('overall'), 'reason': o.get('reason', ''),
            'options': o.get('options', {}), 'diag': diag}


if __name__ == '__main__':
    src, clipdir, out = sys.argv[1:4]
    model = os.environ.get('RB_MODEL', 'gemini-3.1-pro-preview')
    workers = int(os.environ.get('RB_WORKERS', '6'))
    qs = [json.loads(l) for l in open(src)]
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                d = json.loads(l)
                if d.get('overall'):
                    done.add(d['id'])
            except Exception:
                pass
    todo = [q for q in qs if q['id'] not in done]
    print(f'{model}: 共 {len(qs)}, 已核验 {len(done)}, 待核 {len(todo)}', flush=True)
    import time
    t0 = time.time()
    with open(out, 'a') as f:
        for i in range(0, len(todo), 40):
            part = todo[i:i + 40]
            with ThreadPoolExecutor(workers) as ex:
                res = list(ex.map(lambda q: verify(q, clipdir, model), part))
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            n = i + len(part)
            el = time.time() - t0
            ok = sum(1 for r in res if r.get('overall'))
            print(f'  {n}/{len(todo)}  {el:.0f}s  eta {el/max(1,n)*(len(todo)-n):.0f}s  本批成功 {ok}/{len(part)}',
                  flush=True)
    print('done ->', out, flush=True)
