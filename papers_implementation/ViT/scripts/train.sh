#!/usr/bin/env bash
# train ViT model. run from the repository root or ViT directory.
# for instance: bash scripts/train.sh --sanity_check

set -e
cd "$(dirname "$0")/.."
python3 -m training.train "$@"
