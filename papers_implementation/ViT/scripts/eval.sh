#!/usr/bin/env bash
# evaluate ViT checkpoint. run from the repository root or ViT directory (anyone works).
# for instance, see: python3 -m training.eval --checkpoint runs/ckpt.pt


set -e
cd "$(dirname "$0")/.."
python -m training.eval "$@"
