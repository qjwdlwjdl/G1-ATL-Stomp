#!/bin/bash
# Fine-tune SONIC on a local/bare-metal GPU box (no cloud orchestration).
#
# This is the exact recipe that produced the released v8 policy: warm-start from
# the stage-V6 checkpoint and refine for 2,000 iterations at a low actor learning
# rate, on the repaired v7 motion, with the training terminations relaxed.
#
# Why each override is here:
#   actor_learning_rate=4e-6      the policy peaks around iteration 4,000 at the
#                                 stock 1e-5 and degrades afterwards, so the last
#                                 stage is a low-LR refinement rather than a
#                                 longer run.
#   tracking_anchor_pos weight/std   root-tracking emphasis (V1.1b tuning).
#   terminations.*                relaxed from the stock
#                                 tracking/base_adaptive_strict_ori_foot_xyz
#                                 (0.15 m pelvis / 0.2 rad, adaptive tightening)
#                                 to 0.5 m / 1.2 rad / 0.4 m body with adaptive
#                                 off. The stock values end an episode the moment
#                                 the policy starts to struggle, so it never
#                                 learns the recovery region.
#
# Usage:
#   bash code/train_local.sh <start-checkpoint.pt> <experiment-dir> [num-envs] [iters]
# Example:
#   bash code/train_local.sh /opt/sonic/ckpts/v6_4000/model_step_004000.pt \
#                            /opt/sonic/logs/v8_local 2048 2000
set -euxo pipefail

CKPT=${1:?usage: train_local.sh <start-checkpoint.pt> <experiment-dir> [num-envs] [iters]}
EXP_DIR=${2:?usage: train_local.sh <start-checkpoint.pt> <experiment-dir> [num-envs] [iters]}
NUM_ENVS=${3:-2048}
ITERS=${4:-2000}

SONIC=${SONIC:-/opt/sonic}
MOTION=${MOTION:-$SONIC/data/atl_stomp_v7.pkl}
VENV=${VENV:-/opt/venv311}

source "$VENV/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
[ -f "$MOTION" ] || { echo "missing motion: $MOTION"; exit 1; }
mkdir -p "$EXP_DIR"

cd "$SONIC"
# one process per GPU; for multiple GPUs use --num_processes=<n> and
# `accelerate launch --multi_gpu`
accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py \
  "+exp=manager/universal_token/all_modes/sonic_release" \
  "num_envs=$NUM_ENVS" \
  "headless=True" \
  "use_wandb=false" \
  "experiment_dir=$EXP_DIR" \
  "+checkpoint=$CKPT" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=zeros" \
  "++algo.config.num_learning_iterations=$ITERS" \
  "++algo.config.actor_learning_rate=4e-6" \
  "++manager_env.rewards.tracking_anchor_pos.weight=1.0" \
  "++manager_env.rewards.tracking_anchor_pos.params.std=0.2" \
  "++manager_env.terminations.anchor_pos.params.threshold=0.5" \
  "++manager_env.terminations.anchor_pos.params.down_threshold=0.5" \
  "++manager_env.terminations.anchor_pos.params.threshold_adaptive=False" \
  "++manager_env.terminations.ee_body_pos.params.threshold=0.4" \
  "++manager_env.terminations.ee_body_pos.params.down_threshold=0.4" \
  "++manager_env.terminations.ee_body_pos.params.threshold_adaptive=False" \
  "++manager_env.terminations.anchor_ori_full.params.threshold=1.2" \
  "++manager_env.terminations.foot_pos_xyz.params.threshold=0.4"
