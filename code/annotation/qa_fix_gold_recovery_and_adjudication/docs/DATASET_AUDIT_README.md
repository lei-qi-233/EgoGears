# CVHCI video-understanding data audit

Audit date: 2026-08-24

## Sources

- Google Drive controlled running/walking set:
  `https://drive.google.com/drive/folders/1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz`
- Hugging Face reference set:
  `https://huggingface.co/datasets/Mikerog1/CVHCI_MMLM_Shaky_Videos`

The two sources should remain separate domains in metadata. Do not merge and
randomly split them as if all files were independent samples.

## Google Drive set

Expected design: 54 long first-person videos in a complete factorial grid:

- 3 participants: `P01`, `P02`, `P03`
- 3 routes: indices 1, 2, 3 (route correspondence across participants was
  visually verified; P02 calls these `Trial`, while P01/P03 call them `Traj`)
- 3 movement conditions: `FastWalk`, `SlowWalk`, `Running`
- 2 lighting conditions: `Day`, `Night`

Each route therefore has 18 paired recordings. The videos are egocentric
outdoor navigation recordings, not third-person human-action videos.

Current local status: 39/54 videos downloaded (52.24 GB, 8.76 hours). All 39
open successfully, are 1920x1080 HEVC, and have no filename parse failures.
Google Drive stopped serving the remaining 15 files because its shared-file
download quota was exceeded. See `metadata/missing_google_drive_files.txt`.

Important audit findings:

- P01 has audio, P02 does not, and downloaded P03 videos have audio. Audio
  presence is therefore a perfect participant/device shortcut.
- Downloaded P03 files contain precise location and capture-time metadata.
  Derived manifests intentionally retain only boolean risk flags, not values.
- P01/P02 appear to be re-encoded; P03 retains camera metadata and has visibly
  different exposure/color/FOV. Participant and capture-device domains are
  confounded.
- Day/night and speed variants revisit the same routes. Random clip or video
  splitting would cause severe scene leakage.
- Motion blur, camera shake, low-light noise, and wide-angle distortion are
  core properties of the data, not isolated corruptions.

## Hugging Face set (public metadata only)

The repository is manual-gated and this machine has no approved Hugging Face
login, so annotation contents could not be read. Public repository metadata
shows:

- 53.09 GB, 211 files total
- 75 video paths, 94 JSON files, and 41 GPX files
- three contributors/domains: Alec (31 videos), Markus (30), Mikail (14)
- 71 `qa_pairs.json` files plus 23 chunk-annotation/caption JSON files
- three exact duplicate video path pairs by filename and byte size
- two unique video stems without a matching QA stem, and one debug QA stem
  without a matching video stem
- some Alec media live under `Alec/Alec/...` while related metadata is under
  `Alec/...`, so sibling-path joins are unsafe
- no dataset card or declared license in the public Hub metadata

Before using this set, obtain approved access, inspect the actual JSON schema,
hash-deduplicate media, repair path joins, and establish license/consent terms.

## Recommended benchmark design

This collection is strongest as a controlled robustness and cross-video
reasoning benchmark. Recommended evaluations:

1. Leave-one-participant-out (three folds) for device/person generalization.
2. Leave-one-route-out (three folds) for unseen-place generalization.
3. Train on day and test on night for illumination shift.
4. Cross-speed route retrieval/localization: match the same place across
   walking/running speeds.
5. Temporal QA with evidence timestamps: turns, crossings, landmark order,
   entering/leaving regions, and before/after relations.

For any ordinary train/validation/test split, assign groups before making
clips. At minimum group by `(participant, route_index)`; for strict unseen-place
evaluation, hold out the route index across all participants and conditions.
Never randomly split adjacent windows.

Suggested derived views:

- 8-16 second clips for motion/scene tasks, non-overlapping by default
- 30-60 second clips for temporal reasoning
- full videos for route retrieval and long-context evaluation
- remove stationary setup/takedown segments, but preserve turns and genuine
  motion blur

Every annotation should include temporal evidence. Reject questions answerable
from filenames or a single arbitrary frame when the target is video
understanding. Use hard negatives from the same route or lighting condition to
reduce background shortcuts.

## Recommended record schema

Required fields:

- `id`, `source_dataset`, `source_video_id`, `relative_path`
- `participant`, `route_index`, `activity`, `lighting`
- `start_s`, `end_s`, `duration_s`, `split`, `split_group`
- `caption`, `events[{start_s,end_s,label}]`
- `qa[{question,answer,evidence_intervals,question_type}]`
- `quality{brightness,blur,motion_magnitude,shake_score}`
- `provenance`, `license`, `consent_status`, `privacy_review_status`

For public release, strip container metadata, remove or coarsen GPS, review
faces and license plates, and either remove audio from every participant or
define an explicitly audiovisual task with a missing-audio policy.

## Local artifacts

- `raw/google_drive/running_dataset/`: downloaded videos
- `metadata/manifest.jsonl`: safe per-video technical metadata
- `metadata/summary.json`: aggregate statistics
- `metadata/google_drive_listing.json`: expected public file listing
- `metadata/huggingface_public_tree.json`: public HF file-tree snapshot
- `analysis/`: contact sheets used for route/domain checks
- `gemini_pilot/`: two real-video Gemini trials, validation sheets, and audit
- `gemini_annotation_pilot/`: six reviewed Gemini annotation proposals
- `scripts/build_manifest.py`: filename parsing and safe media probing
- `scripts/make_contact_sheet.py`: uniform frame sampler
- `scripts/make_video_clip.py`: metadata-stripping low-bitrate pilot clipper
- `scripts/gemini_video_pilot.py`: structured Floodgate Gemini video call

Resume the Drive download after the quota resets:

```bash
uvx --from gdown gdown --folder --continue \
  'https://drive.google.com/drive/folders/1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz' \
  -O /mnt/data/cvhci_video_understanding/raw/google_drive/
```

Regenerate metadata after all 54 videos are present:

```bash
cd /mnt/data/cvhci_video_understanding
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python av pillow
.venv/bin/python scripts/build_manifest.py \
  raw/google_drive/running_dataset --output-dir metadata
```
