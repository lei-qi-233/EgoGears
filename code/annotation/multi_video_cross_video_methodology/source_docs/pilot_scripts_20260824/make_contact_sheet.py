#!/usr/bin/env python3
"""Extract evenly spaced frames from one or more videos into a contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw


def sample_frames(path: Path, positions: tuple[float, ...]) -> list[tuple[float, Image.Image]]:
    results = []
    with av.open(str(path)) as container:
        duration_s = float(container.duration / av.time_base)
        stream = container.streams.video[0]
        for position in positions:
            target_s = duration_s * position
            timestamp = int(target_s / float(stream.time_base))
            container.seek(timestamp, stream=stream, backward=True)
            frame = next(container.decode(stream))
            results.append((target_s, frame.to_image()))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--positions",
        default="0.1,0.5,0.9",
        help="Comma-separated relative positions from 0 to 1.",
    )
    args = parser.parse_args()

    positions = tuple(float(value) for value in args.positions.split(","))
    if not positions or any(value < 0 or value > 1 for value in positions):
        raise ValueError("positions must be between 0 and 1")
    tile_size = (480, 270)
    label_height = 40
    canvas = Image.new(
        "RGB", (tile_size[0] * len(positions), (tile_size[1] + label_height) * len(args.videos)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row, path in enumerate(args.videos):
        for column, (target_s, frame) in enumerate(sample_frames(path, positions)):
            frame.thumbnail(tile_size)
            x = column * tile_size[0]
            y = row * (tile_size[1] + label_height)
            canvas.paste(frame, (x, y))
            draw.text((x + 6, y + tile_size[1] + 6), f"{path.name} @ {target_s:.1f}s", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=90)


if __name__ == "__main__":
    main()
