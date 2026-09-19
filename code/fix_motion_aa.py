"""Repair the root rotation track of a motion_lib PKL.

Why the free-running policy kept falling: the ATL Stomp clip stores the root
rotation twice - as `root_rot` (a quaternion track) and as `pose_aa[:, 0]`
(axis-angle). The motion library builds the reference pose from `pose_aa`, and
on four frames that track is corrupted: it disagrees with `root_rot` by exactly
120.0 degrees, while agreeing to within 0.06 degrees on every other frame.

   motion frame : 100, 143, 148, 221   (267-frame clip @ 30 fps)
   control frame: 167, 238, 247, 368   (resampled to the 50 Hz control rate)

At each of those frames the reference teleports: every tracked body jumps
10-56 cm in a single 20 ms step while the root does not move. A tracking policy
cannot follow that, and the rollouts die 20-30 control frames later - at frames
190 (v5), 240-247, and 399 (v6) - which is where every free-running render has
been failing.

Fix: rebuild `pose_aa[:, 0]` from the clean `root_rot` quaternions. Quaternion
signs are aligned only where consecutive samples are on opposite hemispheres
(which interpolates along the short arc); the angle is kept over the full
[0, 2*pi] range so a rotation passing through pi - the root spins past 180 deg
between frames 100 and 101 - stays continuous instead of flipping its axis.

Usage: python fix_motion_aa.py <in.pkl> <out.pkl>
"""
import sys

import joblib
import numpy as np


def quat_to_aa(q):
    """Quaternion (xyzw) -> axis-angle, angle kept in [0, 2*pi]."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    xyz, w = q[..., :3], q[..., 3:]
    angle = 2.0 * np.arccos(np.clip(w[..., 0], -1.0, 1.0))
    s = np.sqrt(np.clip(1.0 - w[..., 0] ** 2, 0.0, None))
    axis = xyz / np.where(s < 1e-8, 1.0, s)[..., None]
    out = axis * angle[..., None]
    return np.where((angle < 1e-8)[..., None], 0.0, out)


def aa_to_quat(aa):
    ang = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = (ang[..., 0] < 1e-8)[..., None]
    axis = aa / np.where(ang < 1e-8, 1.0, ang)
    s = np.where(small, 0.5, np.sin(ang / 2))
    q = np.concatenate([axis * s, np.cos(ang / 2)], axis=-1)
    return np.where(small, np.array([0.0, 0.0, 0.0, 1.0]), q)


def ang_between(q1, q2):
    """Rotation angle between two quaternions, in degrees."""
    d = np.abs((q1 * q2).sum(-1))
    return np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0)))


def continuous_aa_from_quat(q):
    """Sign-align a quaternion track, then convert to a continuous aa track."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    for i in range(1, len(q)):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    return quat_to_aa(q)


src, dst = sys.argv[1], sys.argv[2]
d = joblib.load(src)
for name, item in d.items():
    pose_aa = np.asarray(item["pose_aa"]).copy()
    root_rot = np.asarray(item["root_rot"]).astype(np.float64)
    q_ref = root_rot / np.linalg.norm(root_rot, axis=-1, keepdims=True)

    # how wrong is the stored root axis-angle track?
    before = ang_between(aa_to_quat(pose_aa[:, 0, :]), q_ref)
    bad = np.where(before > 5.0)[0]
    print(f"{name}: root track disagrees with root_rot on {len(bad)} frames "
          f"{bad.tolist()} (error {np.round(before[bad], 1).tolist()} deg)")
    print(f"  worst agreement on the other frames: {np.delete(before, bad).max():.3f} deg")

    # rebuild the root column from the clean quaternion track
    pose_aa[:, 0, :] = continuous_aa_from_quat(q_ref).astype(np.float32)

    # the actuated joints get the same sign-alignment treatment as a guard
    for j in range(1, pose_aa.shape[1]):
        pose_aa[:, j, :] = continuous_aa_from_quat(
            aa_to_quat(pose_aa[:, j, :]).astype(np.float64)).astype(np.float32)

    after = ang_between(aa_to_quat(pose_aa[:, 0, :]), q_ref)
    print(f"  after rebuild: max disagreement {after.max():.3f} deg, "
          f"frames >5 deg: {int((after > 5).sum())}")

    q_root = aa_to_quat(pose_aa[:, 0, :])
    step = np.degrees(np.arccos(np.clip(
        np.abs((q_root[1:] * q_root[:-1]).sum(-1)), -1, 1))) * 2.0
    print(f"  root rotation step: median {np.median(step):.2f} deg, "
          f"max {step.max():.2f} deg @f{int(step.argmax())}")

    item["pose_aa"] = pose_aa

joblib.dump(d, dst)
print("saved", dst)
