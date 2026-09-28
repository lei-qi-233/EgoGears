# -*- coding: utf-8 -*-
"""用 Floodgate 上的 Gemini 给题目打题型标签 (9 类报告口径)。"""
import json, os, re, sys, requests
from concurrent.futures import ThreadPoolExecutor

ROOT='https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models'
CERT=('/turibolt_k8s_mounts/narrative/turi/cert.pem','/turibolt_k8s_mounts/narrative/turi/private.pem')

CLASSES=["Object and Attribute QA","Ego-relative Spatial Relation","Trajectory-grounded QA",
 "Ego-motion Recognition","Actor Action and Motion","Scene and Place QA","Temporal Grounding",
 "Event Sequencing","Text and OCR"]

PROMPT="""You classify multiple-choice questions about egocentric outdoor walking/running videos into EXACTLY ONE question-type label.

THE DECISIVE RULE: classify by what DISTINGUISHES the correct option(s) from the distractors — not by the wording of the question stem. The stems are template-generated and carry no signal. Ask: "to answer this, what minimum piece of information must be read off the video?"

LABELS:
1. Text and OCR — must read text/letters/numbers printed on a sign, vehicle or building.
2. Temporal Grounding — must know WHEN something happened (a timestamp/interval). NECESSITY TEST: mentally delete every timestamp from the options; if the correct option is still uniquely identifiable, this is NOT Temporal Grounding.
3. Event Sequencing — the discriminator is ORDER; options are permutations of the same elements.
4. Trajectory-grounded QA — the camera-wearer's own path shape / route across the whole segment.
5. Ego-motion Recognition — one localized maneuver by the camera-wearer (stopping, one turn, zooming, looking down). NOTE: "away from the camera" makes the camera an OBJECT, not the subject — that is not ego-motion.
6. Actor Action and Motion — what an EXTERNAL person/vehicle/animal does or which way it moves.
7. Ego-relative Spatial Relation — the discriminator is DIRECTION relative to the camera or path (left/right/ahead/behind). Distractors name objects that ARE in the video but on the wrong side.
8. Scene and Place QA — place type, weather, time of day, lighting, overall environment.
9. Object and Attribute QA — existence, identity, colour, material, shape, count or state of objects. Use this when distractors name objects that are simply ABSENT from the video, or vary an attribute.

PRIORITY when several fit, first match wins: 1 > 2 > 3 > 4 > 5 > 6 > 7 > 8 > 9.

Answer with JSON only: {"label": "<exact label text>", "why": "<max 15 words naming the discriminator>"}"""

def ask(model,q,retries=3):
    opts='\n'.join(f'  {k}) {v}'+('   <-- CORRECT' if k in q['answer'] else '') for k,v in q['options'].items())
    user=f"QUESTION: {q['question']}\nOPTIONS:\n{opts}\nCORRECT: {', '.join(q['answer'])}"
    s=requests.Session(); s.trust_env=False
    for a in range(retries):
        try:
            r=s.post(f'{ROOT}/{model}:generateContent',
              headers={'X-Floodgate-Project-Token':'','Content-Type':'application/json'},
              json={'systemInstruction':{'parts':[{'text':PROMPT}]},
                    'contents':[{'role':'user','parts':[{'text':user}]}],
                    'generationConfig':{'temperature':0,'maxOutputTokens':4000,
                                        'responseMimeType':'application/json'}},
              cert=CERT,timeout=180)
            if r.status_code!=200: continue
            d=r.json()['candidates'][0]
            t=''.join(p.get('text','') for p in d['content'].get('parts',[]))
            o=json.loads(t)
            lab=o.get('label','')
            if lab in CLASSES: return lab,o.get('why','')
            for c in CLASSES:
                if c.lower() in lab.lower(): return c,o.get('why','')
        except Exception as e:
            if a==retries-1: return None,f'ERR {type(e).__name__}'
    return None,'unparsed'

def run(model,qs,workers=12):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda q:ask(model,q),qs))

def run_resumable(model,src,out,workers=16,chunk=200):
    """断点续跑: 已写入的 id 跳过, 每 chunk 落盘一次"""
    qs=[json.loads(l) for l in open(src)]
    done=set()
    if os.path.exists(out):
        for l in open(out):
            try:
                d=json.loads(l)
                if d.get('label'): done.add(d['id'])
            except: pass
    todo=[q for q in qs if q['id'] not in done]
    print(f'{model}: 共 {len(qs)} 题, 已完成 {len(done)}, 待跑 {len(todo)}',flush=True)
    import time
    t0=time.time()
    with open(out,'a') as f:
        for i in range(0,len(todo),chunk):
            part=todo[i:i+chunk]
            with ThreadPoolExecutor(workers) as ex:
                res=list(ex.map(lambda q:ask(model,q),part))
            for q,(lab,why) in zip(part,res):
                f.write(json.dumps({'id':q['id'],'model':model,'label':lab,'why':why},ensure_ascii=False)+'\n')
            f.flush()
            n=i+len(part); el=time.time()-t0
            print(f'  {model} {n}/{len(todo)}  {el:.0f}s  eta {el/max(1,n)*(len(todo)-n):.0f}s',flush=True)
    print(f'{model}: done -> {out}',flush=True)

if __name__=='__main__':
    run_resumable(sys.argv[1],sys.argv[2],sys.argv[3])
