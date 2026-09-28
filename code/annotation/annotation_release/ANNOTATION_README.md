# RunningBench 人工标注工具包

本目录用于继续 RunningBench 的人工视频问答复核，不包含视频文件。视频应放在 `datasets/GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz/`，并按 `P01/`、`P02/`、`P03/` 保存；程序也会识别旧的 `running_dataset/` 目录。

## 给接手者的快速说明

接手者不需要重新生成题目或重写网页，只需下载本工具包、准备原始视频、修改数据路径，然后启动服务。

### 1. 下载

```bash
mkdir -p ~/runningbench_annotation
cd ~/runningbench_annotation
hf download taryya/RunningBench \
  --repo-type dataset \
  --include "annotation_release/*" \
  --local-dir .
tar -xzf annotation_release/runningbench_annotation_release.tar.gz
cd annotation_release
```

也可以直接使用已经下载的 `annotation_release/` 目录。

### 2. 准备视频

将原始视频放到自己的数据目录，例如：

```text
~/datasets/GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz/P01/*.mp4
~/datasets/GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz/P02/*.mp4
~/datasets/GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz/P03/*.mp4
```

视频不在本工具包内，需要接手者自行获取。视频缺失时，网页会明确提示，不能根据文字猜答案。

### 3. 修改数据路径

打开 `build_review.py`，找到：

```python
b = Path('/home/yuedong_tan/datasets')
```

改成接手者自己的数据根目录，例如：

```python
b = Path('/home/other_user/datasets')
```

该目录下应包含：

```text
review.html
seg600_verification/segments_700_questions_with_category.jsonl
GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz/
```

### 4. 构建和启动

```bash
python3 build_review.py
python3 serve_review.py
```

如果从本地电脑访问远程服务器，另开一个本地终端建立隧道：

```powershell
ssh -N -L 18765:127.0.0.1:18765 user@hostname
```

浏览器打开 `http://127.0.0.1:18765/`。

### 5. 标注和备份

网页支持逐选项 `✓ / ✗ / ?`、最终答案、`clear / ambiguous / cant_tell`、备注、自动保存、JSON 导入和 JSON 导出。建议每完成一批题目就点击“导出 JSON”，把导出的文件单独备份。

### 6. 可选切片

网页默认直接读取原视频并限制在题目时间段内播放，不切片也能标注。如需生成独立题目片段：

```bash
python3 cut_jsonl_videoclips.py \
  --questions segments_700_questions_with_category.jsonl \
  --video-root ~/datasets/GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz \
  --output-root ~/datasets/runningbench_clips
```

切片前先确认原始视频已经完整下载。

## 文件

- `segments_700_questions_with_category.jsonl`：698 道待复核问题。
- `review.html`：原始标注界面模板和标注规则来源。
- `build_review.py`：根据题目和视频建立网页、媒体白名单和覆盖率报告。
- `serve_review.py`：只在本机回环地址提供网页和视频 Range 读取。
- `cut_jsonl_videoclips.py`：可选，把原视频按题目时间段切成独立 MP4。
- `coverage.json`：最近一次构建的媒体覆盖统计。

## 在 hala 上继续

```bash
cd /home/yuedong_tan/datasets/seg600_verification
python3 build_review.py
python3 serve_review.py
```

在本地建立 SSH 隧道后打开网页：

```powershell
ssh -N -L 18765:127.0.0.1:18765 hala
```

浏览器访问 <http://127.0.0.1:18765/>。标注进度保存在浏览器 localStorage；定期点击“导出 JSON”备份。不要凭文字猜测没有视频的题目。

## 补齐视频后的流程

1. 将原视频放到 `datasets/GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz/`。
2. 运行 `python3 build_review.py`，确认 `annotation_app/coverage.json`。
3. 重启 `python3 serve_review.py`，刷新网页。

如果需要独立题目片段，再运行：

```bash
python3 cut_jsonl_videoclips.py \
  --questions segments_700_questions_with_category.jsonl \
  --video-root /home/yuedong_tan/datasets/GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz \
  --output-root /home/yuedong_tan/datasets/runningbench_clips
```

切片前请先确认原视频已经完整下载。当前网页默认直接读取原视频并限制播放到题目时间段，因此切片不是启动网页的前置条件。

## 安全与复现

`serve_review.py` 不提供目录浏览，只允许读取 `media.json` 白名单中的视频；服务绑定 `127.0.0.1`，不要把端口直接暴露到公网。重新构建网页不会覆盖浏览器已有的导出 JSON，导出的 JSON 应单独保存并在必要时通过“导入进度”恢复。
