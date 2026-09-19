# -*- coding: utf-8 -*-
"""Convert 36-column ATL Stomp CSV to SONIC motion_lib PKL.

Layout (per row): root_pos(3) + root_quat_xyzw(4) + 29 G1 joints (MuJoCo order).
150 frames @ 30fps; motion_lib resamples to 50fps at load time.
"""
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R

SRC = "D:/Download/17_atl_stomp_sonic_clean (1).csv"
DST = "data/motions/atl_stomp_fixed.pkl"
FPS = 30.0  # source data fps (matches zip info.txt: source_fps=30)

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

data = np.loadtxt(SRC, delimiter=",")  # (T, 36)
T = data.shape[0]
print(f"loaded {SRC}: {data.shape}")

root_trans_offset = data[:, 0:3].astype(np.float32)
root_quat_xyzw = data[:, 3:7].astype(np.float32)  # (w? verify) xyzw
# sanity: quat w component should be near 1 initially
print("quat col3-6 first row:", root_quat_xyzw[0])

dof = data[:, 7:36].astype(np.float32)  # (T, 29) MJ order
assert dof.shape[1] == 29, f"expected 29 joints, got {dof.shape[1]}"

pose_aa = np.zeros((T, 30, 3), dtype=np.float32)
pose_aa[:, 1:30, :] = DOF_AXIS[None, :, :] * dof[:, :, None]
pose_aa[:, 0, :] = R.from_quat(root_quat_xyzw).as_rotvec()

entry = {
    "root_trans_offset": root_trans_offset,
    "pose_aa": pose_aa,
    "dof": dof,
    "root_rot": root_quat_xyzw,
    "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
    "fps": FPS,
}

import os
os.makedirs("data/motions", exist_ok=True)
joblib.dump({"atl_stomp_fixed": entry}, DST, compress=True)
print(f"saved {DST}: {T} frames @ {FPS}fps")

# quick summary
print("root z first/last:", root_trans_offset[0, 2], root_trans_offset[-1, 2])
print("root y extent:", root_trans_offset[:, 1].min(), root_trans_offset[:, 1].max())
print("joint col range:", dof.min(), dof.max())