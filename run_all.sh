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
#   d1-d9              one change each versus b4_convpass_72ep (d10_full_train, the final model, is run separately)
#   e1-e3              on top of d1_96ep: + d9's stem, 120 epochs, and a second training seed
#   e4-e6              the stem at 120, 150 and 200 epochs (d10_full_train, the final model, is run separately)
#   f1-f5              learning rates and weight decay on e1_stem_96ep; g1 = f5's rates on e6 (200 epochs)
set -e
for config in \
  configs/dinov3_dora_giou.yaml \
  configs/ablations/a1_input.yaml configs/ablations/a2_aug.yaml configs/ablations/a3_wst.yaml \
  configs/ablations/a4_warmstart.yaml configs/ablations/a5_fusion.yaml configs/ablations/a6_warmstart_cascade.yaml \
  configs/ablations/b1_full_blocks.yaml configs/ablations/b2_rank_early.yaml configs/ablations/b3_rank_uniform.yaml \
  configs/ablations/b4_convpass.yaml configs/ablations/b5_hybrid.yaml \
  configs/ablations/c1a_b4_warmstart.yaml configs/ablations/c1b_b4_warmstart_cascade.yaml \
  configs/ablations/b4_convpass_58ep.yaml configs/ablations/c2_rank64_convpass192.yaml \
  configs/ablations/b4_convpass_72ep.yaml configs/ablations/b4_wst_72ep.yaml \
  configs/ablations/d1_96ep.yaml configs/ablations/d2_ema.yaml configs/ablations/d3_head512_cp352.yaml \
  configs/ablations/d4_convhead_cp352.yaml configs/ablations/d5_depthwise640.yaml configs/ablations/d6_multiscale.yaml \
  configs/ablations/d7_parallel.yaml configs/ablations/d8_rpn06.yaml configs/ablations/d9_stem128.yaml \
  configs/ablations/e1_stem_96ep.yaml configs/ablations/e2_120ep.yaml configs/ablations/e3_d1_seed1.yaml \
  configs/ablations/e4_stem_120ep.yaml configs/ablations/e5_stem_150ep.yaml configs/ablations/e6_stem_200ep.yaml \
  configs/ablations/f1_adapter_lr3e-4.yaml configs/ablations/f2_adapter_lr1e-3.yaml configs/ablations/f3_lr2e-4.yaml \
  configs/ablations/f4_wd_exclude_1d.yaml configs/ablations/f5_split_lr.yaml configs/ablations/g1_split_lr_200ep.yaml; do
  python train.py --config "$config" "$@"
done
