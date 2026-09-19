# G1 ATL Stomp — Teaching a Humanoid Robot a Street-Dance Move

**Track:** Performance Arts
**Project name:** G1 ATL Stomp

## What we taught it

The Unitree **G1 humanoid (29 DOF)** performing the **ATL Stomp** — a ~5-second
street-dance routine: stomp footwork, a single-leg hop, a deep squat with a torso
roll, and fast arm swings, non-stop at tempo — followed by a settle back to a
clean neutral stance.

**Stock SONIC fails this move** — in the paired rollout below it loses balance
and falls (5.06 s in), ending on the floor. The released policy performs the
whole routine in free dynamics without falling and finishes standing at rest.

| Metric | Stock SONIC | **final policy** |
|---|---|---|
| Fall onset | 5.06 s | **none** |
| Root height, minimum | 0.077 m (on the floor) | **0.654 m** |
| Tracking error, mean / max | 2901 / 4643 mm | **475 / 753 mm** |

Both rolled out on the same reference (the routine plus the settle tail,
550 control steps @ 50 Hz) in free dynamics with every termination except
`time_out` disabled, so nothing is masked by an episode reset. While dancing,
the reference keeps the root between 0.655 m and 0.822 m.

**Sim video (before / after):** <https://youtu.be/cT_J6KtBf6g>
**Full-length clip:** `videos/v8_slowmotion_50fps.mp4` — the same policy on the
1.5× slowed reference plus the settle tail, all 544 control steps without a
fall: root height minimum 0.663 m, mean tracking error 96 mm, and it ends
standing at rest.

## Why it's hard

- **The move turns.** The routine rotates the body through large yaw changes
  while the feet are planted or mid-stomp. Rotating the base without losing the
  foot contact that is holding it up is the hardest thing to learn here, and it
  is the first thing that fails.
- **Everything moves at once.** Stomp footwork flows straight into a hop and then
  into a squat with a torso roll; arms and legs are never in a neutral phase, and
  an error in one transition propagates into the next.
- **The stomps are impacts.** Each one injects a large ground reaction that has to
  be absorbed without dropping the reference footwork.
- **The training setup hides the failure.** The stock terminations end an episode
  the moment the policy deviates, so the training reward looks healthy while the
  free-running robot is falling over.

## How we solved "it never learns it"

The policy failed in the same place no matter how long we trained, while the
training reward kept improving. The problem was the training objective, not the
policy.

1. **Relaxed the terminations.** The stock set ends an episode once the pelvis
   drifts 0.15 m or rotates 0.2 rad from the reference. During a spin that fires
   the instant the policy starts to struggle, so PPO was reset out of the
   recovery region every time and never saw it. Relaxing to 0.5 m / 1.2 rad /
   0.4 m body (adaptive tightening off) moved the fall from 43 % to 90 % of the
   clip within 500 iterations.
2. **Refined instead of over-training.** The first fine-tune (2,000 iterations on
   the repaired motion) still fell in its own free-dynamics run (3.54 s).
   Training longer was not the answer — the policy peaked near iteration 4,000
   and had degraded again by 6,000 — so the final stage warm-starts from that
   peak and runs 2,000 iterations at a low actor LR (4e-6).
3. **Repaired the reference motion.** The root rotation track disagreed with the
   clean quaternion track by exactly 120° on 4 of 267 frames, teleporting every
   tracked body 10–56 cm in a single 20 ms step. Rebuilt it from the quaternion
   track (`code/fix_motion_aa.py`).
4. **Gave it an ending.** The choreography freezes mid-pose on one leg with the
   arms overhead — the robot holds it, but every viewer reads it as an imminent
   fall. We extended the reference itself (`code/add_settle_tail.py`): hold the
   final pose, ease back to the neutral stance, hold. The released policy
   tracks the settle (mean error 96 mm over the extended clip), so the demo
   ends standing at rest instead of frozen mid-pose.

Every stage was gated by a headless free-dynamics rollout
(`code/verify_rollout.py`) that reports a fall-onset frame or
`COMPLETED_UPRIGHT` — never by training reward.

## Links

| Artifact | Link |
|---|---|
| GitHub repo — code and training config | <https://github.com/qjwdlwjdl/G1-ATL-Stomp> |
| ONNX policy — Hugging Face | <https://huggingface.co/oniichan521/g1-atl-stomp-policy> |
| Dataset — Hugging Face | <https://huggingface.co/datasets/oniichan521/g1-atl-stomp-motion> |
| Sim video — the move, before and after | <https://youtu.be/cT_J6KtBf6g> |
| Trained checkpoint (`.pt`) | GitHub release [`ckpt-v8-2000`](../../releases/tag/ckpt-v8-2000) |

## Repository contents

```
├── code/
│   ├── convert_clean_csv.py        # dance capture CSV → SONIC motion_lib PKL
│   ├── fix_motion_aa.py            # rebuild the root rotation track from root_rot
│   ├── add_settle_tail.py          # append the settle-to-stance tail to a motion PKL
│   ├── train_local.sh              # fine-tune launcher (PPO, relaxed terminations)
│   ├── verify_rollout.py           # headless free-dynamics rollout → fall report
│   ├── render_sim_video.py         # Isaac Sim 5.1 RTX simulation-video renderer
│   ├── make_before_after_video.py  # side-by-side comparison video with narration
│   └── export_onnx.sh              # ONNX export
├── config/
│   ├── training_config.yaml        # resolved training config of the released checkpoint
│   ├── training_config.md          # reproduction commands
│   ├── onnx_README.md              # Hugging Face model card
│   └── dataset_README.md           # Hugging Face dataset card
└── videos/
    ├── atl_stomp_before_after_en.mp4   # base model vs final policy, narrated
    └── v8_slowmotion_50fps.mp4         # the released policy, full-length clip
```

## Environment

- Ubuntu 24.04, Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2
  (GR00T-WholeBodyControl / SONIC)
- Training: NVIDIA L40S (48 GB), 2,048 parallel environments
- Rendering: NVIDIA L40S, Isaac Sim RTX offscreen cameras

## Credit

**Motion Data by Bones Studio.**

Base model: SONIC (NVIDIA GEAR-SONIC, Apache-2.0).
