# G1 ATL Stomp — Teaching a Humanoid Robot a Street-Dance Move

**Track:** Performance Arts
**Project name:** G1 ATL Stomp

## What we taught it

We taught the Unitree **G1 humanoid (29 DOF)** a complete **ATL Stomp** — a
~5-second street-dance routine with rapid stomp footwork, a single-leg hop, a
deep squat with torso roll, and fast arm swings, danced continuously at tempo.
(We also train on a 1.5× slowed copy of the clip; see step 4.)

**The stock SONIC base model fails this move.** It holds balance for the first
two seconds, then loses the reference and falls at 2.92 s. Getting a policy that
actually dances the routine took three separate fixes, described below. The
released policy performs **the entire move in free dynamics without falling**.

| Metric | Stock SONIC | first fine-tune | **final policy** |
|---|---|---|---|
| Fall onset | 2.92 s | 3.54 s | **none** |
| Root height, minimum | 0.086 m (on the floor) | 0.149 m | **0.654 m** |
| Tracking error, mean / max | 598 / 1409 mm | 733 / 1962 mm | **248 / 464 mm** |

All three numbers come from rolling the **same reference motion** out in free
dynamics with every episode termination except `time_out` disabled, so nothing is
masked by an episode reset. The reference keeps the root between 0.655 m and
0.822 m, so a 0.654 m minimum means the policy is on its feet for the whole clip.

**Sim video (before / after):** `videos/atl_stomp_before_after_en.mp4` — stock
SONIC against the final policy, both on the original-tempo clip (248 control
steps, 5.0 s), with a live root-height plot.
**Full-length clip:** `videos/v8_slowmotion_50fps.mp4` — the final policy on the
1.5× slowed reference, all 444 control steps (443 frames @ 50 fps, 1920×1080,
Isaac Sim RTX).

## Why it's hard

- **Far outside the base model's distribution.** Stock SONIC (NVIDIA's whole-body
  control foundation model for the G1) has never seen dance-style motion. On this
  clip it holds balance for about two seconds and then falls.
- **Fast, chained full-body coordination.** Stomp footwork runs straight into a
  single-leg hop and then a deep squat with a torso roll, at dance tempo. Every
  transition needs precise whole-body timing, and an error in one propagates into
  the next.
- **Balance under repeated impacts.** Each stomp injects a large ground impact
  that the controller has to absorb while still holding the reference footwork.
- **The training setup hides the failure.** The stock terminations end an episode
  as soon as the robot deviates, so the training reward looks healthy while the
  free-running policy is falling over. Getting a policy that actually performs the
  move required finding and fixing that, which is most of the work below.

## How we did it

1. **Motion data.** A street-dance capture was converted into a 29-DOF G1 joint
   trajectory and written to SONIC's `motion_lib` PKL format
   (`code/convert_clean_csv.py`).
2. **V2 — behaviour fine-tune.** PPO (TRL) from the official SONIC release
   checkpoint, 2,000 iterations, 2,048 parallel environments. Tracking improves
   substantially over the base model.
3. **V3 — root-tracking refinement.** Another 2,000 iterations emphasising root
   tracking (`tracking_anchor_pos.weight` 0.5 → 1.0, `std` 0.3 → 0.2, actor LR
   2e-5 → 1e-5). It scored well in the official evaluation harness and **still
   fell in free dynamics** — which is what put us onto the real problem.
4. **V5 — time-stretch.** The clip was resampled to 1.5× duration so each motion
   frame gets more control steps. The fall moved later in wall-clock time, which
   pointed away from speed being the whole problem. This is the 8.9 s variant.
5. **V6 — fixing the training objective.** This was the real bottleneck. The stock
   terminations (`tracking/base_adaptive_strict_ori_foot_xyz`) end an episode once
   the pelvis drifts 0.15 m or rotates 0.2 rad from the reference. On a single
   dynamic motion that fires the instant the policy starts to struggle, so PPO
   never sees — and never learns — the recovery region. Relaxing the thresholds
   (0.5 m / 1.2 rad / 0.4 m body, adaptive tightening off) moved the fall from
   **43 % to 90 %** of the clip within 500 iterations.
6. **V8 — refine at the peak.** At the stock 1e-5 learning rate the policy peaked
   near iteration 4,000 (90 % of the clip) and had degraded back to 43 % by
   iteration 6,000. The final stage warm-starts from that peak and refines for
   2,000 iterations at actor LR 4e-6.
7. **Reference-motion repair.** `pose_aa[:, 0]` (the root rotation track)
   disagreed with the clean `root_rot` quaternion by exactly **120°** on 4 of 267
   frames, while agreeing to within 0.05° on every other frame. Each of those
   frames teleports the reference — every tracked body jumps 10–56 cm in a single
   20 ms step — and the policy falls 20–30 control frames later.
   `code/fix_motion_aa.py` rebuilds that column from `root_rot`.
8. **Baseline for comparison.** The stock SONIC release checkpoint is rolled out
   on the same reference with the same settings, so the comparison is
   apples-to-apples: it holds balance for about two seconds and then falls.
9. **Release.** The final policy is exported to ONNX and rendered with Isaac Sim
   5.1 RTX offscreen cameras (`code/render_sim_video.py`). Every stage was gated
   by a headless free-dynamics rollout (`code/verify_rollout.py`) that reports
   either a fall-onset frame or `COMPLETED_UPRIGHT`, rather than by training
   reward.

## Repository contents

```
├── code/
│   ├── convert_clean_csv.py    # dance capture CSV → SONIC motion_lib PKL
│   ├── fix_motion_aa.py        # rebuild the root rotation track from root_rot
│   ├── train_local.sh          # fine-tune launcher for a local GPU box (PPO, relaxed terminations)
│   ├── verify_rollout.py       # headless free-dynamics rollout → fall report
│   ├── render_sim_video.py     # Isaac Sim 5.1 RTX simulation-video renderer
│   ├── make_before_after_video.py  # side-by-side comparison video with narration
│   └── export_onnx.sh          # ONNX export (eval_agent_trl.py +export_onnx_only=true)
├── config/
│   ├── training_config.yaml    # resolved training config of the released checkpoint
│   ├── training_config.md      # reproduction commands (data → train → verify → render → export)
│   ├── onnx_README.md          # Hugging Face model card
│   └── dataset_README.md       # Hugging Face dataset card
└── videos/
    ├── atl_stomp_before_after_en.mp4   # base model vs final policy, narrated
    └── v8_slowmotion_50fps.mp4         # the released policy, full clip
```

## Submission links

| Artifact | Link |
|---|---|
| ONNX policy | Hugging Face: [`oniichan521/g1-atl-stomp-policy`](https://huggingface.co/oniichan521/g1-atl-stomp-policy) |
| Dataset | Hugging Face: [`oniichan521/g1-atl-stomp-motion`](https://huggingface.co/datasets/oniichan521/g1-atl-stomp-motion) |
| Sim video (before / after) | YouTube: *(link)* |

## Environment

- Ubuntu 24.04, Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2
  (GR00T-WholeBodyControl / SONIC)
- Training: Nebius bare metal, NVIDIA L40S (48 GB), 2,048 parallel environments
- Rendering: Nebius bare metal, NVIDIA L40S, RTX offscreen cameras

## Credit

**Motion Data by Bones Studio.**

Base model: SONIC (NVIDIA GEAR-SONIC, Apache-2.0).
