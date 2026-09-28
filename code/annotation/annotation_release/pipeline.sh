#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VIDEO_ROOT="${VIDEO_ROOT:-$ROOT/../GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz}"

python3 "$ROOT/build_review.py"
echo "Coverage:"
cat "$ROOT/annotation_app/coverage.json"
exec python3 "$ROOT/serve_review.py"
