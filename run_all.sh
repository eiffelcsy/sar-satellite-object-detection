#!/usr/bin/env bash
# The base recipe and its ablations, one after another, each driven by its YAML config.
# Extra arguments are passed to every run, e.g. a quick end-to-end check:
#   bash run_all.sh --limit 64 --epochs 1 --out runs/smoke
#
#   dinov3_dora_giou   the base (current best)
#   a1-a5              one change each versus the base; a6 = Cascade on top of a4 (compare with a4)
#   b1-b5              capacity: each builds on a1 (unfrozen patch embedding) and fills ~38-39.5 M of the 40 M
#                      trainable budget a different way (compare with a1)
#   c1a/c1b, c2        combinations of the winners; b4_convpass_58ep is the 58-epoch control for c1a/c1b
#   b4_*_72ep          longer training of the best model, with and without the WST stem
set -e
for config in \
  configs/dinov3_dora_giou.yaml \
  configs/ablations/a1_input.yaml configs/ablations/a2_aug.yaml configs/ablations/a3_wst.yaml \
  configs/ablations/a4_warmstart.yaml configs/ablations/a5_fusion.yaml configs/ablations/a6_warmstart_cascade.yaml \
  configs/ablations/b1_full_blocks.yaml configs/ablations/b2_rank_early.yaml configs/ablations/b3_rank_uniform.yaml \
  configs/ablations/b4_convpass.yaml configs/ablations/b5_hybrid.yaml \
  configs/ablations/c1a_b4_warmstart.yaml configs/ablations/c1b_b4_warmstart_cascade.yaml \
  configs/ablations/b4_convpass_58ep.yaml configs/ablations/c2_rank64_convpass192.yaml \
  configs/ablations/b4_convpass_72ep.yaml configs/ablations/b4_wst_72ep.yaml; do
  python train.py --config "$config" "$@"
done
