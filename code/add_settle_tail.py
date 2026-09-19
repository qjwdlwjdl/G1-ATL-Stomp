# -*- coding: utf-8 -*-
"""Append a settle-to-stance tail to the ATL Stomp motion pkls.

The choreography ends frozen in a stylized one-leg pose. Judges read that as an
imminent fall, so we extend the reference itself: ease from the final frame to
the G1 nominal stance (dof -> 0, pelvis -> standing height, keep heading, level
roll/pitch) over EASE frames with smoothstep, then hold HOLD frames.

Builds pose_aa directly from dof axes + root_rot rotvec (same construction as
convert_clean_csv.py), so the appended frames carry no representation wraps.

Usage: python tools/add_settle_tail.py
"""
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R

DOF_AXIS = np.array(  # G1 29-DOF axes (g1_29dof_rev_1_0.xml), MJ order
    [
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],   # left leg
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],   # right leg
        [0, 0, 1], [1, 0, 0], [0, 1, 0],                                    # waist
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # left arm
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # right arm
    ],
    dtype=np.float32,
)

STAND_Z = None  # resolved per-file from its own first frame (nominal stance height)


def build_tail(entry, ease_n, hold_n, freeze_n=0):
    rt, rr, dof = entry["root_trans_offset"], entry["root_rot"], entry["dof"]
    T = rt.shape[0]

    # yaw-only upright target at the final heading
    Rm = R.from_quat(rr[-1]).as_matrix()
    yaw = np.arctan2(Rm[1, 0], Rm[0, 0])
    q_to = R.from_euler("z", yaw)
    from scipy.spatial.transform import Slerp
    slerp = Slerp([0.0, 1.0], R.from_quat(np.stack([rr[-1], q_to.as_quat()])))

    z_from, z_to = rt[-1, 2], rt[0, 2]  # back to this file's own standing height
    xy_last = rt[-1, :2].copy()
    dof_last = dof[-1].copy()

    # optional freeze: repeat the final dance frame so landing momentum dies
    # before the ease starts
    n = freeze_n + ease_n + hold_n
    t = np.arange(ease_n) / max(ease_n - 1, 1)
    s = (3 * t**2 - 2 * t**3)[:, None].astype(np.float32)  # smoothstep ease

    ease_dof = ((1 - s) * dof_last[None]).astype(np.float32)
    tail_dof = np.vstack([
        np.tile(dof_last[None], (freeze_n, 1)).astype(np.float32),
        ease_dof,
        np.zeros((hold_n, 29), np.float32),
    ])

    ease_rt = np.tile(np.concatenate([xy_last, [z_from]])[None], (ease_n, 1)).astype(np.float32)
    ease_rt[:, 2] = z_from + s[:, 0] * (z_to - z_from)
    tail_rt = np.vstack([
        np.tile(np.concatenate([xy_last, [z_from]])[None].astype(np.float32), (freeze_n, 1)),
        ease_rt,
        np.tile(np.concatenate([xy_last, [z_to]])[None].astype(np.float32), (hold_n, 1)),
    ])

    u = np.concatenate([np.zeros(freeze_n), t, np.ones(hold_n)])
    tail_rr = slerp(u).as_quat().astype(np.float32)

    # pose_aa: joints from dof axes, root from quat rotvec
    tail_pose = np.zeros((n, 30, 3), np.float32)
    tail_pose[:, 1:30, :] = DOF_AXIS[None] * tail_dof[:, :, None]
    tail_pose[:, 0, :] = R.from_quat(tail_rr).as_rotvec().astype(np.float32)

    out = {
        "root_trans_offset": np.vstack([rt, tail_rt]).astype(np.float32),
        "pose_aa": np.concatenate([entry["pose_aa"], tail_pose], 0).astype(np.float32),
        "dof": np.vstack([dof, tail_dof]).astype(np.float32),
        "root_rot": np.vstack([rr, tail_rr]).astype(np.float32),
        "smpl_joints": np.zeros((T + n, 24, 3), np.float32),
        "fps": entry["fps"],
    }
    return out


def report(tag, entry, orig_T):
    dof, rt, rr = entry["dof"], entry["root_trans_offset"], entry["root_rot"]
    T = dof.shape[0]
    jd = np.abs(np.diff(dof, axis=0)).sum(-1)
    e = R.from_quat(rr).as_euler("xyz", degrees=True)
    print(f"{tag}: {orig_T}->{T} frames | last dance step {jd[orig_T-2]:.2f} | "
          f"junction step {jd[orig_T-1]:.2f} | tail step max {jd[orig_T-1:].max():.2f} | "
          f"end z {rt[-1,2]:.3f} | end rpy {np.round(e[-1],1)} | end |dof| {np.abs(dof[-1]).max():.3f}")


if __name__ == "__main__":
    jobs = [
        ("data/motions/atl_stomp_v7.pkl", "data/motions/atl_stomp_v8_tail.pkl", 45, 15, 0),
        ("data/motions/atl_stomp_fixed.pkl", "data/motions/atl_stomp_fixed_tail.pkl", 120, 30, 30),
    ]
    for src, dst, ease_n, hold_n, freeze_n in jobs:
        d = joblib.load(src)
        key, entry = next(iter(d.items()))
        orig_T = entry["dof"].shape[0]
        out = {key: build_tail(entry, ease_n, hold_n, freeze_n)}
        joblib.dump(out, dst, compress=True)
        report(f"{dst} (key {key}, freeze {freeze_n} ease {ease_n} hold {hold_n})", out[key], orig_T)
        print("saved", dst)
