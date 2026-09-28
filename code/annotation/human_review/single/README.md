---
pretty_name: RunningBench Annotation Packs
language:
- en
- zh
tags:
- video
- annotation
- benchmark
- egocentric-video
---

# RunningBench Annotation Packs

This repository contains ten self-contained offline packages for human verification of 698 RunningBench multiple-choice video questions.

## What is included

- 698 question-specific 480p MP4 clips, split across ten archives.
- Parts 01–09 contain 70 questions each; part 10 contains 68 questions.
- Every archive contains:
  - `index.html` — offline annotation interface.
  - `clips/*.mp4` — one video clip per question.
  - `manifest.json` — question metadata without reference answers.
  - `README.md` — package-level instructions.
- `index.json` records archive sizes and SHA-256 checksums.

Reference answers are intentionally excluded from the distributed packages.

## Annotation workflow

1. Download one `runningbench_clips_partXX.tar.gz` archive.
2. Extract the archive completely. Do not open `index.html` from inside an archive viewer.
3. Open `index.html` in Chrome or Edge.
4. Enter the annotator name and review every question.
5. Click **导出 JSON / Export**.
6. Return the generated `runningbench_partXX_annotation_<annotator>_<date>.json` file.

Progress is saved automatically in the browser. Each package uses a separate progress key, so multiple parts can be annotated on the same computer.

## Integrity

Use the SHA-256 values in `index.json` to verify downloaded archives before distribution.

## Source dataset

The broader RunningBench project is available at https://huggingface.co/datasets/taryya/RunningBench.
