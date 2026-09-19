---
license: cc-by-4.0
tags:
- robotics
- motion-capture
- humanoid
- unitree-g1
- locomotion
- dataset
size_categories:
- n<1K
---

# G1 ATL Stomp — Motion Dataset

An **ATL Stomp** street-dance motion clip for the **Unitree G1 humanoid
(29 DOF)**, formatted for the **SONIC / GR00T-WholeBodyControl** `motion_lib`
training pipeline, together with the time-stretched and repaired variants used
for fine-tuning.

## Content

| File | Frames @ 30 Hz | Description |
|---|---|---|
| `atl_stomp_fixed.pkl` | 150 | Original motion clip (joblib PKL, motion_lib format) |
| `atl_stomp_fixed_smooth.pkl` | 150 | Representation-smoothed variant (axis-angle / joint-angle sequences re-aligned to the shortest path) |
| `atl_stomp_v4.pkl` | 178 | Re-cut clip used for the v4 fine-tune |
| `atl_stomp_v5.pkl` | 267 | **1.5× time-stretch of v4** (root rotation slerped, joints linear) — more control steps per motion frame; the reference the V6 and V8 stages were trained on |
| `atl_stomp_v7.pkl` | 267 | **v5 with the root rotation track repaired** (see below) — the reference used for the released policy |
| `atl_stomp_fixed_tail.pkl` | 330 | **fixed + settle-to-stance tail** (freeze 1 s, ease 4 s, hold 1 s) — the reference for the before/after comparison |
| `atl_stomp_v8_tail.pkl` | 327 | **v7 + settle tail** — the reference for the full-length slowed clip |

## Root-track repair (v7)

`v5` stores the root rotation twice: as `root_rot` (a quaternion track) and as
`pose_aa[:, 0]` (axis-angle). The motion library builds the reference pose from
`pose_aa`, and on that track four frames are corrupted:

| | |
|---|---|
| Disagreement with `root_rot` on those frames | **exactly 120.0°** |
| Disagreement on every other frame | ≤ 0.05° |
| Frames (267-frame clip @ 30 Hz) | 100, 143, 148, 221 |
| Corresponding control frames @ 50 Hz | 167, 238, 247, 368 |

At each of those frames the reference teleports — every tracked body jumps
10–56 cm in a single 20 ms step while the root stays put — and a tracking policy
destabilised by it falls 20–30 control frames later. `atl_stomp_v7.pkl` rebuilds
`pose_aa[:, 0]` from the clean `root_rot` quaternions (sign-aligned along the
short arc, angle kept over the full `[0, 2π]` range so a rotation passing
through π stays continuous). Maximum disagreement afterwards: 0.05°.

The repair script is `code/fix_motion_aa.py` in
[`qjwdlwjdl/G1-ATL-Stomp`](https://github.com/qjwdlwjdl/G1-ATL-Stomp).

## Settle tail (`*_tail`)

The choreography ends frozen mid-pose on one leg with the arms overhead. The
tail extends the reference: hold the final pose, a smoothstep ease back to the
neutral stance (dof → 0, pelvis → standing height, heading kept, roll/pitch
levelled), then a hold, so a policy that tracks it ends the clip standing
instead of freezing mid-pose. Built by `code/add_settle_tail.py`; the appended
frames are constructed directly from the dof axes + root quaternion, so they
introduce no representation wraps.

| File | Tail |
|---|---|
| `atl_stomp_fixed_tail.pkl` | 30 frames freeze + 120 frames ease + 30 frames hold (1 s + 4 s + 1 s) |
| `atl_stomp_v8_tail.pkl` | 45 frames ease + 15 frames hold (1.5 s + 0.5 s) |

## Motion specification

- **Clip:** ATL Stomp street-dance routine — stomp footwork, single-leg hop,
  deep squat with torso roll, fast arm swings — ending in a settle back to a
  neutral stance on the `*_tail` variants
- **Skeleton:** Unitree G1, 29 DOF (`dof`), 30-body axis-angle pose (`pose_aa`),
  root translation/rotation (`root_trans_offset`, `root_rot`)
- **Source rate:** 30 Hz; the motion library resamples to the 50 Hz control rate
  at load time (267 frames → 444 control steps for v5/v7)
- **Quality:** no NaN/Inf; joint limits respected; physically consistent

## PKL structure

```python
import joblib
data = joblib.load("atl_stomp_v7.pkl")
# data["atl_stomp_v5"] = {
#   "root_trans_offset": (267, 3)     float32   # root translation
#   "root_rot":          (267, 4)     float32   # root quaternion (xyzw)
#   "pose_aa":           (267, 30, 3) float32   # body axis-angle poses
#   "dof":               (267, 29)    float32   # G1 joint angles (MuJoCo order)
#   "smpl_joints":       (267, 24, 3) float32   # unused placeholder
#   "fps":               30.0
# }
```

## Provenance

Motion captured in-house from a user-provided street-dance video; no third-party
motion data used. Conversion from a 36-column CSV (root position + root
quaternion + 29 G1 joint angles, MuJoCo order) to `motion_lib` format is done by
`code/convert_clean_csv.py`.

## Credit

**Motion Data by Bones Studio.**

## License

Released under CC-BY-4.0.
