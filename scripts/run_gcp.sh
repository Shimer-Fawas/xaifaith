#!/usr/bin/env bash
# Sharded attribution run on a GCP A100 VM.
#
# You only need this if you scale past ~5k molecules x 4 architectures. The
# 4090 handles the default configuration (4 datasets x 4 architectures x 600
# molecules) in a few hours; the A100 is for the full-dataset version.
#
# Provision:
#   gcloud compute instances create xaifaith-a100 \
#     --zone=asia-southeast1-c \
#     --machine-type=a2-highgpu-1g \
#     --accelerator=type=nvidia-tesla-a100,count=1 \
#     --maintenance-policy=TERMINATE \
#     --image-family=common-cu124-ubuntu-2204-py310 \
#     --image-project=deeplearning-platform-release \
#     --boot-disk-size=200GB
#
# The deep-learning image already has CUDA and torch, so on the VM run
# `pip install torch_geometric -e .` rather than the full setup_env.sh.
#
# Cost note: a2-highgpu-1g is roughly $3-4/hour on-demand. Use a spot instance
# (--provisioning-model=SPOT) for the attribution stage - it checkpoints per
# shard, so a preemption costs you at most one shard.

set -euo pipefail

OUT="${OUT:-runs/full}"
NUM_SHARDS="${NUM_SHARDS:-4}"
DATASETS="${DATASETS:-bace bbbp clintox tox21_nr_ar tox21_sr_mmp sider_hepato esol lipo}"
ARCHS="${ARCHS:-gin gat gcn attentivefp}"

# shellcheck disable=SC2086
python -m xaifaith.run train --datasets $DATASETS --architectures $ARCHS --out "$OUT"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  # shellcheck disable=SC2086
  python -m xaifaith.run attribute \
    --datasets $DATASETS \
    --architectures $ARCHS \
    --out "$OUT" \
    --max-molecules 0 \
    --shard "$shard" \
    --num-shards "$NUM_SHARDS" &
done
wait

python -m xaifaith.run meta --out "$OUT"

# Pull the results back:
#   gcloud compute scp --recurse xaifaith-a100:~/xaifaith/runs/full ./runs/ --zone=asia-southeast1-c
# And stop the instance the moment it finishes:
#   gcloud compute instances stop xaifaith-a100 --zone=asia-southeast1-c
