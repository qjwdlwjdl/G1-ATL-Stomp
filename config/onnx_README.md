---
license: apache-2.0
base_model: nvidia/GEAR-SONIC
tags:
- robotics
- whole-body-control
- humanoid
- unitree-g1
- onnx
- RL
- fine-tuning
library_name: onnx
---

# G1 ATL Stomp — ONNX Whole-Body Control Policy

Fine-tuned **SONIC** whole-body control policy for the **Unitree G1 humanoid (29 DOF)**
that performs **ATL Stomp** — a ~5-second street-dance routine with rapid stomp
footwork, a single-leg hop phase, a deep squat with torso roll, and fast arm
swings.

The stock SONIC base model fails this move: it holds balance for the first two
seconds, then loses the reference and falls at 2.92 s. The released policy
completes **the whole routine in free dynamics without falling**, on both the
original-tempo clip and the 1.5× slowed variant it was trained on.

## Results

Free-dynamics rollout of the original-tempo clip (248 control steps @ 50 Hz),
every episode termination except `time_out` disabled, so nothing is masked by an
episode reset:

| Metric | Stock SONIC | first fine-tune | **this release** |
|---|---|---|---|
| Fall onset | 2.92 s | 3.54 s | **none** |
| Root height, minimum | 0.086 m (on the floor) | 0.149 m | **0.654 m** |
| Tracking error, mean / max | 598 / 1409 mm | 733 / 1962 mm | **248 / 464 mm** |

The same policy on the 1.5× slowed reference runs all 444 control steps with a
root-height minimum of 0.663 m and a mean tracking error of 92 mm.

See `atl_stomp_before_after_en.mp4` in the companion GitHub repo
([`qjwdlwjdl/G1-ATL-Stomp`](https://github.com/qjwdlwjdl/G1-ATL-Stomp)) for a
side-by-side render of both policies on the same reference.

## Files

| File | Description |
|---|---|
| `model_step_002000_g1.onnx` | G1 policy (actor) — primary deployment artifact |
| `model_step_002000_encoder.onnx` | observation encoder |
| `model_step_002000_decoder.onnx` | action decoder |
| `model_step_002000_smpl.onnx` | SMPL encoder head |
| `model_step_002000_teleop.onnx` | teleop encoder head |

Exported with the official `gear_sonic` pipeline
(`eval_agent_trl.py +export_onnx_only=true`).

## Training

- **Base model:** [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)
  (SONIC release checkpoint, universal-token whole-body controller).
- **Method:** PPO (TRL) motion tracking on a single street-dance clip, 2,048
  parallel environments in Isaac Lab.
- **Stage V8** (this release): 2,000 iterations, actor LR 4e-6, warm-started from
  stage V6 and trained on the repaired `atl_stomp_v7` motion.

Three findings drove the final stage, each verified by rollout rather than by
training reward:

1. **Termination thresholds, not the policy, were the bottleneck.** The stock
   training terminations (`tracking/base_adaptive_strict_ori_foot_xyz`) end an
   episode once the pelvis drifts 0.15 m or rotates 0.2 rad from the reference.
   On a single dynamic motion that fires the moment the policy starts to
   struggle, so PPO never sees — and never learns — the recovery region.
   Relaxing them (0.5 m / 1.2 rad / 0.4 m body, adaptive tightening off) moved
   the fall from 43 % to 90 % of the slowed clip within 500 iterations.
2. **Long runs regress.** At the stock 1e-5 learning rate the policy peaked
   around iteration 4,000 (90 % of the slowed clip) and had degraded back to
   43 % by iteration 6,000. The final stage therefore warm-starts from the peak and
   refines at 4e-6.
3. **The reference motion had a corrupted track.** `pose_aa[:, 0]` (root
   rotation) disagreed with the clean `root_rot` quaternion by exactly 120° on
   4 of 267 frames, while agreeing to within 0.05° everywhere else. Each of
   those frames teleports the reference (every tracked body jumps 10–56 cm in a
   single 20 ms step). `atl_stomp_v7` rebuilds that column from `root_rot`.

## Usage

The policy is compatible with the GR00T-WholeBodyControl deployment stack
(Unitree G1, 29-DOF joint interface). Observations follow the SONIC
universal-token policy interface (proprioception + motion-tracking command
features at 50 Hz; actions are 29-DOF joint position targets).

## Training data

Motion tracking on the in-house **ATL Stomp** motion clip — see the companion
dataset repo
([`g1-atl-stomp-motion`](https://huggingface.co/datasets/oniichan521/g1-atl-stomp-motion)).
Motion data captured in-house from a user-provided street-dance video; no
third-party motion data used.

## Credit

**Motion Data by Bones Studio.**

## License

Fine-tuned weights released under Apache-2.0. Base model: NVIDIA GEAR-SONIC
(Apache-2.0).
