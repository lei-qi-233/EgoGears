# -*- coding: utf-8 -*-
"""Floodgate 调用的公共层: mTLS + 429 退避 + JSON 兜底解析。

踩过的坑:
- 限流是 HTTP 429, 不是异常。原来的 `if status!=200: continue` 会把 3 次重试瞬间烧完,
  结果整批返回 unparsed 而看起来像"解析失败"。必须退避重试并尊重 Retry-After。
- session.trust_env=False: 环境里的代理会弄坏 mTLS 握手。
- 即使设了 responseMimeType=application/json, 模型仍可能裹围栏或裹成数组。
"""
import json, os, random, re, threading, time, requests

ROOT = 'https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models'
CERT = ('/turibolt_k8s_mounts/narrative/turi/cert.pem',
        '/turibolt_k8s_mounts/narrative/turi/private.pem')

_local = threading.local()
_gate = threading.Semaphore(int(os.environ.get('RB_CONC', '4')))


def _session():
    s = getattr(_local, 's', None)
    if s is None:
        s = requests.Session()
        s.trust_env = False
        _local.s = s
    return s


def clean_json(raw):
    d = None
    try:
        d = json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw or '', re.S)
        if m:
            try:
                d = json.loads(m.group(0))
            except Exception:
                d = None
    if isinstance(d, list):
        d = next((x for x in d if isinstance(x, dict)), None)
    return d if isinstance(d, dict) else None


def generate(model, parts, system=None, timeout=180, max_tokens=8000, tries=8):
    """返回 (dict|None, 诊断字符串)。max_tokens 给够: thinking token 也吃这个预算。"""
    body = {'contents': [{'role': 'user', 'parts': parts}],
            'generationConfig': {'temperature': 0, 'maxOutputTokens': max_tokens,
                                 'responseMimeType': 'application/json'}}
    if system:
        body['systemInstruction'] = {'parts': [{'text': system}]}
    last = 'no attempt'
    for a in range(tries):
        try:
            with _gate:
                r = _session().post(f'{ROOT}/{model}:generateContent',
                                    headers={'X-Floodgate-Project-Token': '',
                                             'Content-Type': 'application/json'},
                                    json=body, cert=CERT, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get('Retry-After')
                wait = float(ra) if ra and ra.isdigit() else min(60, 2 ** a) * (1 + random.random())
                last = f'HTTP{r.status_code}'
                time.sleep(wait)
                continue
            if r.status_code != 200:
                return None, f'HTTP{r.status_code} {r.text[:120]}'
            j = r.json()
            c = j['candidates'][0]
            txt = ''.join(p.get('text', '') for p in c.get('content', {}).get('parts', []))
            o = clean_json(txt)
            if o:
                return o, c.get('finishReason', '')
            last = f'unparsed finish={c.get("finishReason")} len={len(txt)}'
        except Exception as e:
            last = f'{type(e).__name__}: {str(e)[:80]}'
            time.sleep(min(30, 2 ** a) * (1 + random.random()))
    return None, last
