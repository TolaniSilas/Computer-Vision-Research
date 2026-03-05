#!/usr/bin/env bash
# train resnet-18 model from config. run from the repository root or ViT directory (anyone works).
# for instance, see: python3 -m training.train --config config/default.yaml


set -e
cd "$(dirname "$0")/.."
python3 -m training.train --config config/default.yaml "$@"
