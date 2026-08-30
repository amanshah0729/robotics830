#!/usr/bin/env bash
# End-to-end smoke test on synthetic data — no flash drive, no torch needed.
# From repo root:  bash scripts/fixture_demo.sh   then open http://localhost:7860
set -euo pipefail
cd "$(dirname "$0")/.."

python -m musclememory.ingest --source fixture --embedder fake
python -m musclememory.index --derived work/derived --out work/index --proj pca
python -m musclememory.eval --derived work/derived
python -m musclememory.server --source fixture --embedder fake --port 7860
