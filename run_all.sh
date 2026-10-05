#!/usr/bin/env bash
# The eight reference runs, one after another, each driven by its YAML config.
# Extra arguments are passed to every run, e.g. a quick end-to-end check:
#   bash run_all.sh --limit 64 --epochs 1 --out runs/smoke
set -e
# Pretrained backbones in the `full` configs are fine-tuned gently (train.backbone_lr 2e-5 in the YAML);
# adapters and scratch backbones use the default 1e-4.
for config in \
  configs/vit_full.yaml configs/vit_lora.yaml configs/vit_moelora.yaml \
  configs/terramind_full.yaml configs/terramind_lora.yaml configs/terramind_moelora.yaml \
  configs/vit_scratch.yaml configs/terramind_scratch.yaml; do
  python train.py --config "$config" "$@"
done
