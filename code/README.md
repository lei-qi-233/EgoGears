# Code

This directory contains the code, annotation tools, evaluation harnesses, verification scripts, and supporting documentation used to build and evaluate the EgoGears benchmark.

> **Repository status:** The code files in this directory have not been fully organized yet. The current structure is only a preliminary classification into `annotation` and `evaluation`. File names, code comments, and the English Markdown documentation will continue to be revised and updated.

The materials are organized into two areas:

- [`annotation/`](annotation/) — question annotation, human review, visual verification, question repair, and annotation methodology;
- [`evaluation/`](evaluation/) — single-video and multi-video evaluation harnesses, model analysis, and evaluation releases.

## Directory Structure

```text
code/
├── annotation/
│   ├── annotation_release/
│   ├── human_review/
│   ├── multi_video_cross_video_methodology/
│   ├── qa_fix_gold_recovery_and_adjudication/
│   ├── segments_60s_gemini_verification/
│   └── ANNOTATION_PROCESS.md
└── evaluation/
    ├── model_evaluation_harness/
    └── single_video_567/
```

## Annotation

The [`annotation/`](annotation/) directory contains the workflows used to create, review, verify, repair, and release benchmark questions.

- [`ANNOTATION_PROCESS.md`](annotation/ANNOTATION_PROCESS.md) describes the differences between single-video and multi-video annotation, including clip ordering, evidence verification, distractor design, and quality gates.
- [`human_review/`](annotation/human_review/) contains the human annotation interface, review manuals, and annotation bundles.
- [`annotation_release/`](annotation/annotation_release/) contains release-oriented annotation scripts, review pages, coverage reports, and question files.
- [`multi_video_cross_video_methodology/`](annotation/multi_video_cross_video_methodology/) documents the methodology and annotation workflow for multi-video and cross-video questions.
- [`qa_fix_gold_recovery_and_adjudication/`](annotation/qa_fix_gold_recovery_and_adjudication/) contains question repair, gold-answer recovery, adjudication, documentation, and final data materials.
- [`segments_60s_gemini_verification/`](annotation/segments_60s_gemini_verification/) contains model-assisted verification materials for 60-second video segments.

Human review is performed at the option level. Annotators check whether each option is supported, ruled out, or undecidable based on the video evidence, then provide a final answer and a clarity verdict. For multi-clip questions, clip labels do not necessarily represent chronological order.

## Evaluation

The [`evaluation/`](evaluation/) directory contains model evaluation harnesses, result analysis, and evaluation releases.

### Model Evaluation Harness

[`model_evaluation_harness/`](evaluation/model_evaluation_harness/) contains evaluation harnesses, frame extraction, scoring, model result aggregation, and analysis for multiple vision-language models.

### Single-Video Evaluation Release

[`single_video_567/`](evaluation/single_video_567/) is a portable single-video evaluation release containing 567 human-curated questions. The associated video clips are required separately and are not stored in this repository. Its [`README.md`](evaluation/single_video_567/README.md) documents the release contents, data requirements, evaluation protocol, frame extraction settings, scoring rules, and reproducibility checklist.

The included single-video evaluation protocol uses one video clip per question, deterministic option shuffling, a fixed frame budget, strict JSON answer parsing, and exact, overlap, and well-formedness metrics.

### Multi-Video Evaluation

The annotation and evaluation materials together cover questions that require evidence integration across clips, correspondence across independent recordings, route-phase reasoning, travel-direction comparison, and ordered route-state tracking. Refer to the README inside each relevant subdirectory before running its scripts.

## Reproducibility Notes

The subdirectories are maintained as separate workflows and may have different dependencies, input data, model backends, and runtime assumptions. Before running code:

1. Read the README or runbook in the relevant subdirectory.
2. Check the expected data paths and required video files.
3. Verify model names, backend settings, frame extraction parameters, and output locations.
4. Preserve the generated metadata and per-question outputs needed for later scoring and analysis.

This directory includes both source code and intermediate or release-supporting materials. Do not assume that every file is part of one end-to-end command or that every referenced source path is available in this repository.