"""根据问题 JSONL 文件批量生成 480p H.264 视频片段。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ==================== 可直接修改的配置 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
INPUT_JSONL = SCRIPT_DIR / "segments_700_questions_with_category.jsonl"
RAW_VIDEO_DIR = WORKSPACE_DIR / "raw_videos"
OUTPUT_DIR = SCRIPT_DIR / "videoclips"
OUTPUT_HEIGHT = 480
DEFAULT_CRF = 23
DEFAULT_PRESET = "medium"
DEFAULT_OVERWRITE = False
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 JSONL 裁切视频，输出 480p H.264 MP4 片段。"
    )
    parser.add_argument("--jsonl", type=Path, default=INPUT_JSONL, help="问题 JSONL 文件路径")
    parser.add_argument("--raw-videos", type=Path, default=RAW_VIDEO_DIR, help="原始视频目录")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="视频片段输出目录")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF, help="H.264 质量参数，数值越小质量越高")
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="libx264 编码速度预设")
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE, help="覆盖已经存在的片段")
    return parser.parse_args()


def load_records(jsonl_path: Path) -> list[dict]:
    """读取 JSONL，并检查每条记录的必需字段和时间范围。"""
    records = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            for field in ("question_id", "video_id", "start_second", "end_second"):
                if field not in record:
                    raise ValueError(f"Missing '{field}' on line {line_number}")
            start = float(record["start_second"])
            end = float(record["end_second"])
            if start < 0 or end <= start:
                raise ValueError(f"Invalid time range on line {line_number}: {start} to {end}")
            record["start_second"] = start
            record["end_second"] = end
            records.append(record)
    return records


def index_source_videos(raw_videos: Path) -> dict[str, Path]:
    """递归扫描原始视频，并建立 video_id 到文件路径的索引。"""
    index: dict[str, Path] = {}
    for path in raw_videos.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        stem = path.stem
        if stem in index:
            raise ValueError(f"Multiple source videos have the same video_id '{stem}'")
        index[stem] = path
    return index


def create_clip(
    source: Path,
    output: Path,
    start: float,
    duration: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> None:
    """调用 ffmpeg 裁切、缩放并编码单个视频片段。"""
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        f"scale=-2:{OUTPUT_HEIGHT}",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    """校验输入后，逐条生成以 question_id 命名的视频片段。"""
    args = parse_args()
    if not args.jsonl.is_file():
        print(f"JSONL file not found: {args.jsonl}", file=sys.stderr)
        return 1
    if not args.raw_videos.is_dir():
        print(f"Raw video directory not found: {args.raw_videos}", file=sys.stderr)
        return 1

    records = load_records(args.jsonl)
    sources = index_source_videos(args.raw_videos)
    missing = sorted({record["video_id"] for record in records} - sources.keys())
    if missing:
        print("Missing source videos:", file=sys.stderr)
        for video_id in missing:
            print(f"  {video_id}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    for number, record in enumerate(records, start=1):
        start = record["start_second"]
        duration = record["end_second"] - start
        output = args.output / f"{record['question_id']}.mp4"
        # 默认跳过已有文件，避免重复编码；需要重新生成时使用 --overwrite。
        if output.exists() and not args.overwrite:
            print(f"[{number}/{len(records)}] skip existing {output.name}")
            continue
        print(f"[{number}/{len(records)}] {output.name} from {record['video_id']}")
        create_clip(
            sources[record["video_id"]],
            output,
            start,
            duration,
            args.crf,
            args.preset,
            args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())