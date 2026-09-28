# 用 Floodgate 调 Gemini 做视频标注 —— 完整流程

本文从 RunningBench 已跑通的代码里倒出来，所有常量、提示词、参数都能在恢复树里找到出处。

**本文的代码是验过的，不是抄的**（2026-09-12 在本机）：

- §2 的客户端和 §5 的 `clean_json` 直接从本文抽出来执行，两个都能用。
- 视频那条路径整条走通了：拿恢复出来的 `P01_Running_Day_Traj01.mp4`，
  按 §4.2 的 ffmpeg 参数切 12 秒（1.7 MB），inline 送进 `gemini-3.1-pro-preview`，
  返回 `{"setting": "urban", "surface": "paved", "motion": "running"}` —— 判对了。
- §3 里 pro 模型 thinking token 吃预算那条，是这次实测出来的新结论。

代码位置：

| 用途 | 文件 |
|---|---|
| 纯文本调用 + 盲猜闸门 | `blind_gate_repair_20260906/gate_segments_60s.py` |
| 视频调用（inline 上传） | `s3://yuedong/backups/task_runtime_code_5hk3xk6dxy_20260909/recheck_segments_60s.py` |
| 整片标注主流程（五道闸门） | `bundles/runningbench_handoff/04_脚本/build_fullvideo_annotations.py` |
| 单次视频标注最小样例 | `scripts/gemini_video_pilot.py` |

> ⚠️ **已知缺口**：`build_fullvideo_annotations.py` 靠 `load_generator()` 动态加载
> `repair_runningbench_annotations.py`，从那里取 `Floodgate`、`CAPTION_MODEL`、
> `STRUCTURE_MODEL`、`clean_json`、`generate_valid_json`。**这个文件没有备份到 Conductor**，
> 恢复树里没有，所以主流程目前跑不起来（全文引用了它 15 处）。
> 它的等价实现分散在 `gate_segments_60s.py`（文本）和 `recheck_segments_60s.py`（视频）里，
> 按本文第 2 节重建一个模块即可补上。

---

## 1. 认证与端点

Floodgate 认的是**挂载的客户端证书**，不是 token。

```python
FLOODGATE_ROOT = "https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models"
CERT = ("/turibolt_k8s_mounts/narrative/turi/cert.pem",
        "/turibolt_k8s_mounts/narrative/turi/private.pem")
```

请求头里那个 `X-Floodgate-Project-Token` **可以是空字符串**——主流程里就是
`os.environ.get("FLOODGATE_PROJECT_TOKEN", "")`，本机实测传空串照样 HTTP 200。
真正卡住你的永远是证书挂载在不在。

调用形状是标准 Vertex 的 `generateContent`：

```
POST {FLOODGATE_ROOT}/{model}:generateContent
headers: X-Floodgate-Project-Token, Content-Type: application/json
cert:    (cert.pem, private.pem)          ← requests 的 mTLS 参数
```

两个必须设的：

- **`session.trust_env = False`**。不关的话 requests 会去读环境里的 `HTTP(S)_PROXY`，
  Bolt 环境下代理会把 mTLS 握手弄坏。
- **`timeout`**。文本 180 秒，带视频 600 秒。

## 2. 最小可用客户端

这是 `gate_segments_60s.py` 和 `recheck_segments_60s.py` 里那个类的合并版，
`parts` 既接受纯文本也接受 inline 视频：

```python
import base64, json, os, random, re, threading, time
import requests

FLOODGATE_ROOT = "https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models"
CERT = ("/turibolt_k8s_mounts/narrative/turi/cert.pem",
        "/turibolt_k8s_mounts/narrative/turi/private.pem")


class RateLimiter:
    """全局共享一个，所有线程从同一个令牌桶取，否则并发一开配额立刻超。"""
    def __init__(self, per_second):
        self.interval, self.lock = 1.0 / per_second, threading.Lock()
        self.next_slot = time.monotonic()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_slot - now)
            self.next_slot = max(now, self.next_slot) + self.interval
        if wait:
            time.sleep(wait)


class Floodgate:
    def __init__(self, token, limiter):
        self.token, self.limiter = token, limiter
        self.session = requests.Session()
        self.session.trust_env = False        # 见 §1
        self.lock = threading.Lock()

    def generate(self, model, parts, max_tokens=16384, attempts=5, timeout=600):
        url = f"{FLOODGATE_ROOT}/{model}:generateContent"
        payload = {"contents": [{"role": "user", "parts": parts}],
                   "generationConfig": {"maxOutputTokens": max_tokens,
                                        "temperature": 0.2,
                                        "responseMimeType": "application/json"}}
        headers = {"X-Floodgate-Project-Token": self.token,
                   "Content-Type": "application/json"}
        last = "unknown"
        for k in range(attempts):
            self.limiter.acquire()
            try:
                r = self.session.post(url, headers=headers, json=payload,
                                      cert=CERT, timeout=timeout)
                r.raise_for_status()
                body = r.json()
                texts = [p["text"] for p in body.get("candidates", [{}])[0]
                         .get("content", {}).get("parts", []) if "text" in p]
                if texts:
                    return "\n".join(texts)
                last = f"empty candidates: {str(body)[:200]}"
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
            # 【关键】一串 429 会把这个 session 池里的连接弄成永久不可用，
            # 后面每一次调用都因为跟请求本身无关的原因失败。重建 session，
            # 别把剩下的重试次数花在一个不会恢复的 socket 上。
            if k >= 2:
                with self.lock:
                    try:
                        self.session.close()
                    except Exception:
                        pass
                    self.session = requests.Session()
                    self.session.trust_env = False
            time.sleep(min(60, 4 * (2 ** min(k, 4))) + random.random() * 2)
        raise RuntimeError(last)
```

那个 session 重建不是防御性代码，是实测踩出来的：多小时的跑批中途一旦撞上 429 爆发，
不重建 session 的话**剩下几个小时全部失败**，而且报错信息完全误导人。

## 3. 三个模型，各干各的

| 模型 | 角色 | 用在哪 |
|---|---|---|
| `gemini-3.1-pro-preview` | `CAPTION_MODEL` | 所有**看视频**的调用：整片结构标注、视觉回看闸门 |
| `gemini-3.5-flash` | 出题 / 判定 | 盲猜闸门、干扰项重写 |
| `gemini-3.6-flash` | pilot | 早期单片试标 |

分工的理由：看视频的调用贵且慢（单次约 25 秒），判定类调用量大但便宜。
**盲猜闸门必须和历史数字用同一个模型同一段提示词**，否则新旧盲猜率不可比。

### ⚠️ pro 模型的 thinking token 会吃掉你的输出预算

本机实测（2026-09-12）：

```
maxOutputTokens=256   finishReason=MAX_TOKENS  thoughtsTokenCount=244  text='Here is the JSON requested:\n```'
maxOutputTokens=2048  finishReason=STOP        thoughtsTokenCount=259  text='{\n  "ok": true\n}'
```

`thoughtsTokenCount` 是**算进 `maxOutputTokens` 的**。同一个 9-token 的提示词，
给 256 预算就只剩 8 token 写正文，直接截断；给 2048 才正常返回。

这条很坑，因为上面客户端会把它当成 `empty candidates` 去重试，而**重试解决不了预算不足**——
五次全废，还白白烧了五次配额。`recheck_segments_60s.py` 默认给 16384 就是这个原因。
凡是用 pro，输出预算别低于 4096。

## 4. 视频怎么喂进去

### 4.1 inline base64，不走文件上传

```python
parts = [
    {"text": prompt},
    {"inlineData": {"mimeType": "video/mp4",
                    "data": base64.b64encode(clip.read_bytes()).decode("ascii")}},
]
```

**体积上限**（代码里是硬常量，超了就得重新编码）：

| 场景 | 上限 | 出处 |
|---|---|---|
| 视觉回看 / 片段级 | `20 * 1024 * 1024` | `recheck_segments_60s.py` |
| 整片全局 pass | `11 * 1024 * 1024` | `build_fullvideo_annotations.py` |

注意 base64 会涨 4/3，所以真正的裸字节预算还要再打个折。

### 4.2 编码参数是调出来的，别随手改

```bash
ffmpeg -y -ss <start> -i <source> -t <len> \
  -vf scale=-2:480 -c:v libx264 -preset veryfast -crf 30 -an \
  -movflags +faststart <dst>
```

- **480p 是底线，不能降到 360p**。实测 360p 会让模型把一块清晰可读的路牌报成
  "blurry"，于是判出假的 `contradicted`，把本来正确的题误杀。
- 片子太大**只加 crf，不降分辨率**：`recheck` 里按 `(34, 38)` 逐级重编码直到进 20 MB。
- `-an` 去音轨。P01 有音频、P02 没有、P03 有——音频存在与否是个完美的参与者捷径，
  必须去掉，否则模型能靠"有没有声音"反推是谁录的。
- 整片全局 pass 用 360p/10fps，超过 `GLOBAL_SPLIT_S = 900` 秒就切两半分别送，
  否则单文件塞进 11 MiB 后码率会掉到 70 kbps 以下，什么都看不清。

### 4.3 时间戳：烧进画面，别让模型估

整片标注在**左上角烧了一个 MM:SS 的时钟**，提示词里明写：

> A clock in the TOP-LEFT corner shows the true recording time as MM:SS.
> READ every time you report from that clock and convert to seconds;
> never estimate time from pacing.

不这么做的话模型会按"走了多久感觉"去猜秒数，evidence span 全是错的。

## 5. 返回 JSON 的解析

即使设了 `responseMimeType: "application/json"`，模型**照样会裹 ```json 围栏**——
本机这次冒烟测试里 flash 就返回了 `'Here is the JSON requested:\n```json'`。
所以永远不要直接 `json.loads(raw)`：

```python
def clean_json(raw):
    d = None
    try:
        d = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)          # 从围栏里抠出对象
        if m:
            try:
                d = json.loads(m.group(0))
            except Exception:
                d = None
    if isinstance(d, list):                           # pro 有约一半调用裹成数组
        d = next((x for x in d if isinstance(x, dict)), None)
    if isinstance(d, dict):
        return d
    # 长 notes 会顶穿 maxOutputTokens，JSON 在字符串中间被截断。
    # verdict 是第一个字段，通常还活着——抢救出来并打上 truncated 标记，
    # 别把一个真实判断当成解析失败丢掉。
    m = re.search(r'"verdict"\s*:\s*"(supported|contradicted|insufficient)"', raw or "")
    if m:
        n = re.search(r'"notes"\s*:\s*"(.*)', raw, re.S)
        return {"verdict": m.group(1), "truncated": True,
                "notes": (n.group(1)[:1200] if n else "")}
    return None
```

三种失败模式各对应一段代码：围栏包裹、数组包裹、超长截断。都是实际发生过的。

想更省事就用 **`responseSchema`** 把结构钉死（`gemini_video_pilot.py` 的做法），
传一个 JSON Schema 进 `generationConfig`，模型会按字段填。但围栏问题依然存在，
`clean_json` 还是得留着。

## 6. 标注流程本身

`build_fullvideo_annotations.py` 的五个阶段，评测单位是**完整的一整段录像**，
不切样本（60s/180s 窗口只当内部时间索引用）：

```
1. GLOBAL PASS   整片 360p/10fps → pro → route_phases / turns /
                 return_to_start / revisited landmarks / environment_stages /
                 dynamic_events。粗但完整：要的是顺序和结构。
2. TIME INDEX    已有的 3 分钟窗口观察并进来做 time_index，给 evidence 更细的时间戳。
3. 整片题        每段录像只用它自己的标注出题。
4. 跨片题        同一条路线的所有录像的标注合起来出题。
5. 五道闸门      结构检查 · 泄漏黑名单 · 题干时间戳 · 长度线索 ·
                 高清视觉回看（480p，比全局 pass 更清楚）· 盲猜 ×3 · 确定性选项洗牌
```

### 出题提示词的硬规则（这些是踩出来的，不是想出来的）

- **干扰项必须来自本视频自己的素材**：把自己的 phase/landmark/turn 重新排序、左右镜像、
  挪到错误的 phase、跟错误的 landmark 配对。
- 同路线的**兄弟录像只能提供本视频确实没有的 landmark**。
  `route-summary` / `environment_stages` / `start_end_relation` 三类**绝不能**拿兄弟录像的
  真实描述当干扰项——同一条路线，兄弟的描述在这儿通常也是真的。
- **绝不引入标注里根本不存在的环境或物体**（没有海滩、高速、森林、室内跑道、桥、木栈道）。
  一个没看过视频、只有常识的人必须觉得每个选项一样可信。
- **往返路线**（`facts.turnaround_sec` 存在）的顺序题和左右题必须点明是去程还是回程，
  且所有 evidence span 必须落在 turnaround 的同一侧——去程的左就是回程的右。
- 被砍掉的两类题：`same-route-identification`（9 题只有 1 题可用）和
  `route-retrieval`（20 题只有 2 题可用）。它们问的是**整片级的全称命题**
  （"是不是同一条路线"、"哪些视频包含这个序列"），任何一组 ≤60 秒的窗口都判不了，
  回看器总能在被引用证据之外找到一个岔路或一个匹配，于是把正确答案判成 contradicted。
  换成了"两段视频在**哪里**分岔"和"这一对 landmark 是**什么顺序**"——同样的能力，
  但答案能从窗口本身判定。

## 7. 盲猜闸门

这是整套东西里最重要的一道，因为**不看视频就能答对的题，作为视频理解评测题是无效的**。

提示词（逐字，改了就跟历史数字不可比）：

```python
BLIND_PROMPT = """Answer this multiple-choice question WITHOUT any video — you have none. Return only
JSON: {{"answer": ["..."]}}

QUESTION: {question}
OPTIONS: {options}
{arity}"""
```

判定规则：**三票取二**。`hits >= 2` 即判 `guessable`。
温度 0.2 下三票有 86.5% 完全一致，所以配对实验里降到一票也够用（`--votes 1`），
但报告总体盲猜率时用三票。

### 别把恒答上限当盲猜率

项目里曾经拿 17.6% 的**恒答上限**（永远选同一个字母能得多少分）当盲猜率汇报，
实测盲猜率是 **48.6%**，差了将近 3 倍。这两个是完全不同的量，别混。

### 干扰项修复：v1 为什么卡住，v2 改了什么

v1 的做法是"干扰项必须被证据否定"，结果卡在 73.4% 下不去。这是**设计问题**，不是写得不够好：

> 必须被证据否定 ⟹ 只能写视频里没有的东西 ⟹ 那些东西先验上也更不常见
> ⟹ 正确答案自动成为选项里唯一"像真的"那个。

```
Q: 路两侧紧邻什么？
   灰砾石 / 铁栏杆 / 高灌木 / **绿草** / 砖墙 / 沙土
```

每个干扰项都规规矩矩被证据否定了，答案依然一眼可见——草就是路边最平常的东西。

v2 两处改动：

1. **先验似真性对齐**（关键）：要求每个干扰项是"没看过视频的人会觉得至少同样可能"的东西。
   原则是**常见但不在场**，而不是不常见且不在场。
2. **对抗迭代**：每轮改完立刻盲测，还被猜中就把模型的实际选择回传给改写者要求针对性重写，
   最多 3 轮。v1 是一次性的，从不知道自己输在哪。

轮数分布证明迭代值钱：1 轮成功 88 道、2 轮 34 道、**3 轮 182 道**。只做一轮会丢掉后面 216 道。

### 必须有对照组

修复的对象是"因为被猜中才入选"的题，只测修复后一次会把**回归均值**算成修复的功劳。
实测对照组（同题原版再测一次）是 95.4% 而不是 100%，那 16 个百分点就是回归均值。

| 阶段 | 盲猜可答 | 95% CI |
|---|---:|---|
| 原版（对照） | 95.4% | [93.0, 97.0] |
| v1 修复后 | 73.4% | [69.0, 77.3] |
| **v1 + v2 串联** | **33.1%** | **[28.8, 37.7]** |

### 另一条：不要无条件重写

v1 对所有题无条件重写，把 432 道单选里**本来合格的 13 道改坏了**。
正确做法（`pipeline.py` 的分级）：先盲测原版，过了就原样保留，一次调用都不多花；
没过才进 v1；v1 没救回才进 v2；全失败就**退回原版**并标记 `still_guessable`，
绝不保留一个"唯一证据是它被尝试过"的重写。

## 8. 配额与并发

```python
QUOTA_PER_SECOND = 0.45
```

**是延迟受限，不是配额受限**：单次视频调用约 25 秒，串行只能跑到 0.185 次/秒，
连一半配额都吃不到。用 **12 路并发共享同一个 `RateLimiter`** 才能打满 0.45。

```python
limiter = RateLimiter(QUOTA_PER_SECOND)      # 全局唯一
api = Floodgate(os.environ.get("FLOODGATE_PROJECT_TOKEN", ""), limiter)
with ThreadPoolExecutor(max_workers=12) as pool:
    ...
```

限速器必须是**全局一个**，每个线程自己建一个就等于并发几路配额乘几倍。

## 9. 断点续跑

所有脚本都支持，做法统一：**逐条追加 JSONL 进度文件，启动时先读回来**。

```python
def load(path):
    out = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:      # 进程被杀会留下半行，跳过就好
            continue
        if not r.get("error") and r.get("id"):
            out[r["id"]] = r
    return out
```

写的时候加锁、每条 flush，别攒在缓冲区里——`pipeline.progress.jsonl.corrupt_backup`
就是攒着攒着被杀留下的残骸。

多小时的跑批还要脱离会话：

```bash
setsid nohup python3 pipeline.py --qdir ... > run.log 2>&1 &
```

普通后台会随会话结束被带走。再配一个同样 setsid 的同步守护把产出推回 Conductor
（`blind_gate_repair_20260906/sync_daemon.sh`），否则 `/mnt` 一没就全丢。

## 10. 跑一次

```bash
cd /mnt/task_runtime/blind_test
setsid nohup python3 pipeline.py \
  --qdir "/mnt/data/cvhci_video_understanding/bundles/runningbench_handoff/02_全量数据" \
  --workers 12 --v2-rounds 3 \
  > pipeline.log 2>&1 &
```

`--qdir` 下要有 `questions/segments_60s.jsonl` 和 `captions/segments_60s.jsonl`。

冒烟测试（确认证书和网关通，10 秒）：

```python
import requests
s = requests.Session(); s.trust_env = False
r = s.post("https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models/"
           "gemini-3.5-flash:generateContent",
           headers={"X-Floodgate-Project-Token": "", "Content-Type": "application/json"},
           json={"contents": [{"role": "user", "parts": [{"text": 'Return only JSON: {"ok": true}'}]}],
                 "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.2,
                                      "responseMimeType": "application/json"}},
           cert=("/turibolt_k8s_mounts/narrative/turi/cert.pem",
                 "/turibolt_k8s_mounts/narrative/turi/private.pem"),
           timeout=120)
print(r.status_code, r.json()["candidates"][0]["content"]["parts"])
```

## 11. 验收门槛

新标注的题，**随机抽 10% 做独立盲猜，可猜率必须低于 15%**。
现有语料是 48.6%，这是要改进的目标。

注意按选项数拆开看：4 选项 55.8%（2.2× 随机基线）· 3 选项 67.3% · 2 选项 77.1%。
Video Bench 那批有 444 道只有 2 个选项（随机基线就是 50%），
只看总体数字会得出错误结论。
