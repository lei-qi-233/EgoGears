# -*- coding: utf-8 -*-
"""按 question_type_spec.md v2 的规则对 segments_60s.jsonl 做单标签分类。
判定顺序: D2 -> D1 -> A -> E -> C -> B  (见 spec 第 2 节)"""
import json, re, collections, sys

import os
# 输入路径: 环境变量 RB_SRC > 命令行第一参数 > 默认
SRC=os.environ.get('RB_SRC') or (sys.argv[1] if len(sys.argv)>1 else
    '/mnt/data/cvhci_video_understanding/segments60s_kit/questions/segments_60s.jsonl')
OUT=os.environ.get('RB_OUT','question_types_full_auto.jsonl')

STOP=set(('a an the of on in at to and or with is are was were be been it its this that these those side '
 'left right along near during video camera path which following correct describe describes true visual '
 'details observed seen located from for by as their there also both while when what where how').split())
def toks(s): return {w for w in re.findall(r'[a-z]+',s.lower()) if w not in STOP and len(w)>3}

EGO_OBJ=re.compile(r'\b(away from|towards?|toward|past|behind|in front of|at) the camera\b',re.I)
EGO=re.compile(r'\b(the camera|camera(\'s|’s)? (movement|trajectory|motion|path|operator|person|wearer)|'
 r'camera wearer|the runner|the person|they|it)\b.{0,60}?\b(move|moves|moving|moved|walk|walks|walking|jog|jogs|'
 r'jogging|run|runs|running|turn|turns|turning|stop|stops|stopped|veer|veers|curve|curves|pan|pans|pivot|pivots|'
 r'tilt|tilts|zoom|zooms|travel|travels|continue|continues|proceed|proceeds|cross|crosses|ascend|descend|'
 r'straight|forward|backward|stationary)',re.I)
EGO2=re.compile(r'\b(moves?|walks?|runs?|jogs?|turns?|stops?|veers?|curves?|pans?|travels?|starts? by|continues?|'
 r'remains? (essentially )?stationary|proceeds?)\b',re.I)
ACTOR=re.compile(r'\b(cyclist|pedestrian|runner|jogger|person|people|man|woman|child|dog|car|van|truck|tram|'
 r'bus|scooter|bicycle|vehicle|bird)\b',re.I)
ACTOR_MOVE=re.compile(r'\b(approaches?|passes?|passing|walks?|walking|runs?|running|rides?|riding|drives?|'
 r'driving|reverses?|crosses?|crossing|moves?|moving|flies|flying|enters?|exits?|leaves?|away from|towards? the '
 r'camera|from the (left|right))\b',re.I)
REGION=re.compile(r'\b(left|right)[- ]?(hand )?side\b|directly ahead|directly behind|to the (left|right)\b|'
 r'\b(left|right) of\b|relative to|spatial (relation|arrangement|layout)',re.I)
TIME=re.compile(r'\b\d{1,2}:\d{2}\b|\bat \d{1,2} ?(seconds?|s)\b')
SEQ_Q=re.compile(r'\b(sequence|order|chronological|first.{0,30}then|followed by|after (the|which|it|they)|'
 r'before (the|reaching|it|they))\b',re.I)
OCR=re.compile(r'\b(text|word|words|number|numbers|letter|letters|written|reads?\b|says|displayed|inscription|'
 r'label|lettering|name)\b',re.I)
QUOTED=re.compile(r"['‘’\"]\s*[\w.\- ]{1,30}\s*['‘’\"]")
# --- 时间轴判定 (spec v4 第 4 节) ---
TEXPR=re.compile(r'\b\d{1,2}:\d{2}\b|\bearly(?: on| in)?\b|\bmid-?way\b|\blate(?:r)?\b in the|'
 r'\bat the (?:very )?(?:start|beginning|end)\b|\b(?:initial|final|first|last|closing|opening) '
 r'(?:segment|part|portion|section|half|seconds?|\d+ seconds?)\b|\bfirst \d+ seconds?\b|'
 r'\blast \d+ seconds?\b|\bend of the video\b|\bbeginning of the video\b',re.I)
TWINDOW_Q=re.compile(r'\b(?:during|in|within|throughout) the (?:first|last|final|initial|opening|closing)\b|'
 r'\bfrom 00:\d{2} to 00:\d{2}\b|\b(?:00:\d{2}\s*[-–]\s*00:\d{2})\b|'
 r'\bat the (?:very )?(?:start|beginning|end) of the video\b|\bin the (?:final|initial) segment\b|'
 r'\bbefore (?:you |the camera |they )?reach|\bafter the camera\b|\bafter (?:the|they) (?:turn|turns|makes the turn)\b',re.I)
TRANSITION=re.compile(r'\btransitions? (?:from|into)\b|\bchanges? from\b|\bcomes? into (?:clear )?view\b|'
 r'\bappears?\b|\bdisappears?\b|\bgives way to\b|\bturns? into\b|\bbefore .{0,25}\bthen\b',re.I)

def norm_t(x): return re.sub(r'\s+',' ',x.lower().strip())

def temporal_scope(r):
    """返回 'scoped' (需定位到特定时间窗) 或 'static' (60s 内任一时刻单帧可判)"""
    # 1) 题干限定时间窗
    if TWINDOW_Q.search(r['question']): return 'scoped'
    # 2) 时间表述在选项间取值不同 -> 时间是区分维度 (单一取值不算)
    vals=set()
    hit=0
    for v in r['options'].values():
        m=TEXPR.findall(v)
        if m: hit+=1; vals.update(norm_t(x if isinstance(x,str) else x[0]) for x in m)
    if hit>=2 and len(vals)>=2: return 'scoped'
    # 3) 正确项描述状态变化 -> 必须比较两个时刻
    if any(TRANSITION.search(r['options'][k]) for k in r['answer']): return 'scoped'
    return 'static'

SCENE=re.compile(r'\b(weather|lighting|sky|overcast|sunny|cloudy|daytime|night-?time|dusk|dawn|sunrise|sunset|'
 r'rain|fog|mist|season|environment type|kind of (place|area|setting))\b',re.I)

def load():
    rows=[json.loads(l) for l in open(SRC)]
    for r in rows: r['_seg']=(r['video'],tuple(r['span_sec']))
    return rows

def build_present(rows):
    """同片段其他题的正确选项 -> 已确认存在的描述 (spec 2b)"""
    seg=collections.defaultdict(list); vid=collections.defaultdict(list)
    for r in rows:
        for k in r['answer']:
            t=toks(r['options'][k])
            seg[r['_seg']].append((r['id'],t)); vid[r['video']].append((r['id'],t))
    return seg,vid

def distractor_presence(r,seg,vid):
    """返回干扰项中"在本片段确认存在"的比例; 证据不足返回 None"""
    pool=[t for i,t in vid[r['video']] if i!=r['id']]     # 视频级: 片段级池太稀疏, 信号被压平
    if len(pool)<10: return None
    dis=[k for k in r['options'] if k not in r['answer']]
    hit=n=0
    for k in dis:
        t=toks(r['options'][k])
        if len(t)<2 or len(t)>12: continue   # 多物体拼接选项不可靠, 跳过 (spec 2b 注)
        n+=1
        if any(len(t&p)>=2 and len(t&p)/len(t)>=0.4 for p in pool): hit+=1
    return hit/n if n>=2 else None

def permutation_like(r):
    """选项是同一组元素的不同排列 -> 顺序题"""
    ks=list(r['options']); ts=[toks(r['options'][k]) for k in ks]
    if len(ks)<4: return False
    big=[t for t in ts if len(t)>=4]
    if len(big)<4: return False
    sims=[]
    for i in range(len(big)):
        for j in range(i+1,len(big)):
            u=len(big[i]|big[j]); sims.append(len(big[i]&big[j])/u if u else 0)
    return sorted(sims)[len(sims)//2]>=0.55

def timestamp_necessary(r):
    """必要性测试 (spec 优先级 1): 抹掉时间戳后正确项是否仍唯一"""
    def strip(s): return TIME.sub('',s).strip()
    if not any(TIME.search(v) for v in r['options'].values()): return False
    bare={k:strip(v) for k,v in r['options'].items()}
    for k in r['answer']:
        for k2,v2 in bare.items():
            if k2!=k and v2==bare[k]: return True   # 抹掉后撞车 -> 时间戳必需
    # 抹掉后仍唯一 -> 时间戳不是区分点; 除非题干直接问时刻
    return bool(re.search(r'at (approximately )?what time|at what point|when does|how (long|many seconds)',
                          r['question'],re.I))

def ego_answer(r):
    txt=' '.join(r['options'][k] for k in r['answer'])
    txt_s=EGO_OBJ.sub(' ',txt)          # 去掉 "away from the camera" 等宾语用法
    if EGO.search(txt_s): return True
    txt=txt_s
    # "Starts by moving straight, turns right..." 主语省略
    if EGO2.search(txt) and not ACTOR.search(txt.split(',')[0]): 
        return bool(re.search(r'\b(camera|trajectory|path|walk|move|turn|straight|stationary)\b',r['question'],re.I))
    return False

ACTOR_Q=re.compile(r'what (sequence of )?(actions?|does|do)\b|what does the \w+ (do|perform)|'
 r'\b(cyclist|pedestrian|runner|jogger|car|van|truck|tram|dog)\b.{0,40}\b(do|does|perform|move)',re.I)
LIST_Q=re.compile(r'which of the following .{0,40}(are|is) (visible|present|located|observed|seen|situated)|'
 r'which of the following (objects|details|features|elements|items|visual)',re.I)
def actor_answer(r):
    txt=' '.join(r['options'][k] for k in r['answer'])
    if not (ACTOR.search(txt) and ACTOR_MOVE.search(txt)): return False
    if ACTOR_Q.search(r['question']): return True
    return not LIST_Q.search(r['question'])

def whole_span(r):
    q=r['question']
    return bool(re.search(r'throughout|overall|from (the )?start to (the )?end|from beginning to end|'
                          r'trajectory|movement pattern|sequence of (turns|movements|directions|actions)|what (sequence|series) of|to capture|over the course of',q,re.I))

def classify(r,seg,vid):
    pres=distractor_presence(r,seg,vid)
    # 1. D2
    if timestamp_necessary(r): return 'D2',pres
    # 2. D1
    if permutation_like(r) and SEQ_Q.search(r['question']): return 'D1',pres
    # 3. A
    if ego_answer(r): return ('A2' if whole_span(r) else 'A1'),pres
    # 3b. E
    if actor_answer(r): return 'E1',pres
    # 4. C
    if REGION.search(r['question']) and pres is not None and pres>=0.6: return 'C1',pres
    # 5. B
    if OCR.search(r['question']) and QUOTED.search(' '.join(r['options'].values())): return 'B3',pres
    if SCENE.search(r['question']): return 'B4',pres
    return ('BT' if temporal_scope(r)=='scoped' else 'BS'),pres

# 报告口径 (9 类): BS/BT 合并回 Object and Attribute QA, Text and OCR 独立
REPORT={'BS':'Object and Attribute QA','BT':'Object and Attribute QA',
        'C1':'Ego-relative Spatial Relation','A2':'Trajectory-grounded QA',
        'A1':'Ego-motion Recognition','E1':'Actor Action and Motion',
        'B4':'Scene and Place QA','D2':'Temporal Grounding','D1':'Event Sequencing',
        'B3':'Text and OCR'}

if __name__=='__main__':
    rows=load(); seg,vid=build_present(rows)
    out=[]
    for r in rows:
        c,p=classify(r,seg,vid)
        out.append({'id':r['id'],'video':r['video'],'span_sec':r['span_sec'],'arity':r['arity'],
                    'old':r.get('question_type'),'code':c,'temporal_scope':temporal_scope(r),
                    'report_class':REPORT[c],
                    'distractor_presence':None if p is None else round(p,2)})
    with open(OUT,'w') as f:
        for o in out: f.write(json.dumps(o,ensure_ascii=False)+'\n')
    print('wrote',len(out),'->',OUT)
