#!/usr/bin/env python3
"""Create a privacy-safer, low-bitrate video clip for MLLM pilot calls."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-s", type=float, default=450.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(args.input)) as source, av.open(str(args.output), "w") as destination:
        source_stream = source.streams.video[0]
        destination_stream = destination.add_stream("libx264", rate=args.fps)
        destination_stream.width = args.width
        destination_stream.height = args.height
        destination_stream.pix_fmt = "yuv420p"
        destination_stream.options = {"crf": "28", "preset": "fast"}

        source.seek(
            int(args.start_s / float(source_stream.time_base)),
            stream=source_stream,
            backward=True,
        )
        end_s = args.start_s + args.duration_s
        next_sample_s = args.start_s
        output_index = 0
        output_time_base = Fraction(1, args.fps)
        for frame in source.decode(source_stream):
            if frame.pts is None:
                continue
            frame_s = float(frame.pts * frame.time_base)
            if frame_s < next_sample_s:
                continue
            if frame_s >= end_s:
                break
            converted = frame.reformat(
                width=args.width,
                height=args.height,
                format="yuv420p",
            )
            converted.pts = output_index
            converted.time_base = output_time_base
            for packet in destination_stream.encode(converted):
                destination.mux(packet)
            output_index += 1
            next_sample_s = args.start_s + output_index / args.fps

        for packet in destination_stream.encode():
            destination.mux(packet)

    print(
        f"created={args.output} frames={output_index} "
        f"duration_s={output_index / args.fps:.3f} bytes={args.output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
