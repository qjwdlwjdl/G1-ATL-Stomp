# Training Configuration — G1 ATL Stomp

Full reproduction pipeline, run on a single bare-metal NVIDIA GPU box:
environment → motion data → fine-tune (V2 → V8) → verification → ONNX export →
simulation video.

All training uses the stock SONIC trainer
(`gear_sonic/train_agent_trl.py` from GR00T-WholeBodyControl) driven by
`accelerate launch`; there is no cloud orchestration layer.

## 0. Environment

Bare Ubuntu 24.04 with the NVIDIA driver already installed (the Nebius
"Ubuntu 24.04 for NVIDIA GPUs" image ships driver 580 and the Vulkan ICD):

```bash
python3.11 -m venv /opt/venv311 && source /opt/venv311/bin/activate
pip install --upgrade pip 'setuptools<81' wheel
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone --depth 1 -b v2.3.2 https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab
(cd /opt/IsaacLab/source/isaaclab && pip install -e . --no-deps)
git clone --depth 1 https://github.com/NVlabs/GR00T-WholeBodyControl.git /opt/sonic
(cd /opt/sonic && pip install -e "gear_sonic/[training]" --no-build-isolation)
```

Notes learned the hard way:

- `flatdict==4.0.1` must be installed on its own with `--no-build-isolation`; in a
  bulk `pip install` it fails to build against `setuptools<81`.
- The Vulkan ICD must be pinned (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json`)
  or the RTX renderer picks llvmpipe and fails.
- `hidapi` (pulled in by Isaac Lab) dlopens `libusb-1.0.so.0`, which the base
  image does not ship — `apt-get install -y libusb-1.0-0`.

Hardware used: NVIDIA L40S (48 GB), 8 vCPU, 31 GB RAM.

## 1. Motion data

Source: a street-dance capture as a 36-column CSV (root position + root
quaternion + 29 G1 joint angles, MuJoCo order) at 30 Hz.

```bash
# CSV -> SONIC motion_lib PKL. Edit the SRC/DST paths at the top of the file.
python code/convert_clean_csv.py                 # -> data/motions/atl_stomp_fixed.pkl

# Repair the root rotation track (see README step 7). Required for the V8 stage:
# pose_aa[:, 0] disagrees with the clean root_rot quaternion by exactly 120 deg
# on 4 of 267 frames, which teleports the reference on those frames.
python code/fix_motion_aa.py data/motions/atl_stomp_v5.pkl data/motions/atl_stomp_v7.pkl
```

Stage V5 is a 1.5× time-stretch of the clip (more control steps per motion
frame); V7 is V5 with the root track repaired. Both are in the
[Hugging Face dataset](https://huggingface.co/datasets/oniichan521/g1-atl-stomp-motion).

## 2. Training

Stage V2 (behaviour fine-tune from the SONIC release checkpoint):

```bash
cd /opt/sonic
accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=<released SONIC>/last.pt \
  num_envs=2048 headless=True use_wandb=false \
  experiment_dir=/opt/sonic/logs/v2 \
  ++algo.config.num_learning_iterations=2000 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/opt/sonic/data/atl_stomp_v4.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=zeros
# result: tracking clearly better than the base model (see the table at the end)
```

Stage V3 (root-tracking refinement — tracked well in evaluation but still fell in
free dynamics):

```bash
accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=/opt/sonic/logs/v2/last.pt \
  num_envs=2048 headless=True use_wandb=false \
  experiment_dir=/opt/sonic/logs/v3 \
  ++algo.config.num_learning_iterations=2000 \
  ++algo.config.actor_learning_rate=1e-5 \
  ++manager_env.rewards.tracking_anchor_pos.weight=1.0 \
  ++manager_env.rewards.tracking_anchor_pos.params.std=0.2 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/opt/sonic/data/atl_stomp_v4.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=zeros
# result: scored well in the official eval harness, but still fell in free dynamics
```

Stage V6 (the fix that mattered — relaxed, non-adaptive terminations) and stage
V8 (low-LR refinement from the V6 peak on the repaired V7 motion) are the same
command with different overrides; `code/train_local.sh` packages the V8 recipe:

```bash
bash code/train_local.sh <start-checkpoint.pt> /opt/sonic/logs/v8_local 2048 2000
```

Key hyperparameters:

| Parameter | V2 | V3 | V6 | **V8 (released)** |
|---|---|---|---|---|
| motion | v4 | v4 | v5 | **v7 (repaired)** |
| iterations | 2000 | +2000 | 6000 | **2000** |
| actor_learning_rate | 2e-5 | 1e-5 | 1e-5 | **4e-6** |
| tracking_anchor_pos.weight / std | 0.5 / 0.3 | 1.0 / 0.2 | 1.0 / 0.2 | **1.0 / 0.2** |
| anchor_pos threshold (adaptive) | 0.15 (on) | 0.15 (on) | 0.5 (off) | **0.5 (off)** |
| anchor_ori_full threshold | 0.2 | 0.2 | 1.2 | **1.2** |
| ee_body_pos / foot_pos_xyz | 0.15 / 0.2 | 0.15 / 0.2 | 0.4 / 0.4 | **0.4 / 0.4** |
| starts from | SONIC release | V2 | V3/V5 | **V6 @ iteration 4000** |

## 3. Verification (the gate that matters)

Training reward is a poor proxy: with the stock terminations the reward looks
healthy while the free-running policy is falling over. Every stage is gated by a
headless free-dynamics rollout that disables all terminations except `time_out`
and reports either a fall-onset frame or `COMPLETED_UPRIGHT`:

```bash
source /opt/venv311/bin/activate
cd /opt/sonic
python code/verify_rollout.py \
  --ckpt /opt/sonic/logs/v8_local/last.pt \
  --motion /opt/sonic/data/atl_stomp_v7.pkl \
  --out /tmp/v8_tracks.npz --max-steps 444
```

Needs physics only (no RTX), so it runs on any CUDA box in a couple of minutes.

## 4. ONNX export

```bash
bash code/export_onnx.sh
# writes <ckpt_dir>/exported/model_step_XXXXXX_{g1,smpl,teleop,encoder,decoder}.onnx
```

## 5. Simulation video (Isaac Sim RTX offscreen)

```bash
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json   # avoid llvmpipe

# the submitted before/after, on the original-tempo clip (5.0 s)
python code/render_sim_video.py --ckpt /opt/sonic/ckpts/sonic_release/last.pt \
  --out stock_orig_50fps.mp4 --gpu 0 \
  --motion /opt/sonic/data/atl_stomp_fixed.pkl --max-steps 248 --fps 50
python code/render_sim_video.py --ckpt /opt/sonic/ckpts/v8root/last.pt \
  --out v8_orig_50fps.mp4 --gpu 0 \
  --motion /opt/sonic/data/atl_stomp_fixed.pkl --max-steps 248 --fps 50

# the full-length run on the 1.5x slowed reference (8.9 s)
python code/render_sim_video.py --ckpt /opt/sonic/ckpts/v8root/last.pt \
  --out v8_slowmotion_50fps.mp4 --gpu 0 \
  --motion /opt/sonic/data/atl_stomp_v7.pkl --max-steps 444 --fps 50

# compose the side-by-side comparison
python code/make_before_after_video.py \
  --before stock_orig_50fps.mp4 --before-tracks stock_orig_50fps.mp4.tracks.npz \
  --after v8_orig_50fps.mp4 --after-tracks v8_orig_50fps.mp4.tracks.npz \
  --out atl_stomp_before_after_en.mp4
```


## Results

All three policies rolled out on the **same** reference — the original-tempo clip,
248 control steps at 50 Hz — in free dynamics with every termination except
`time_out` disabled. Reference root height over the clip: 0.655–0.822 m.

| Metric | Stock SONIC | first fine-tune | **final policy** |
|---|---|---|---|
| Fall onset | 2.92 s | 3.54 s | **none** |
| Root height, minimum | 0.086 m (on the floor) | 0.149 m | **0.654 m** |
| Tracking error, mean / max | 598 / 1409 mm | 733 / 1962 mm | **248 / 464 mm** |

On the 1.5x slowed reference the final policy runs all 444 control steps with a
root-height minimum of 0.663 m and a mean tracking error of 92 mm.
