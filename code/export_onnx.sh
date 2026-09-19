#!/bin/bash
# Export the v8 ATL Stomp policy to ONNX for the HuggingFace model repo.
#
# Uses the official gear_sonic path: eval_agent_trl.py +export_onnx_only=true,
# which writes five files per step (g1 / smpl / teleop encoders, plus the
# combined encoder and the action decoder) into {experiment_dir}/exported/.
#
# The checkpoint's saved config.yaml carries the training machine's paths and
# the old `groot.rl.*` module prefix. This script copies the checkpoint into a
# local work dir and rewrites the config so the env builds on this box.
#
# Usage (on the instance, as root): bash /opt/g1stomp/deploy/export_onnx_v8.sh
set -euxo pipefail
source /opt/venv311/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
cd /opt/sonic

SRC=${SRC:-/opt/sonic/ckpts/v8root}
DEST=${DEST:-/opt/sonic/ckpts/v8export}
MOTION=${MOTION:-/opt/sonic/data/atl_stomp_v7.pkl}

rm -rf "$DEST"
mkdir -p "$DEST"
cp -f "$SRC/last.pt" "$SRC/config.yaml" "$DEST/"
[ -f "$SRC/meta.yaml" ] && cp -f "$SRC/meta.yaml" "$DEST/"

python - "$DEST/config.yaml" "$DEST" "$MOTION" <<'PY'
import io
import sys

import omegaconf

cfg_path, dest, motion = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(cfg_path, encoding="utf-8").read()
for s, r in [
    ("groot.rl.trl.", "gear_sonic.trl."),
    ("groot.rl.envs.", "gear_sonic.envs."),
    ("groot.rl.utils.", "gear_sonic.utils."),
    ("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module."),
    ("groot.rl.agents.", "gear_sonic.trl."),
    ("groot/rl/data/", "gear_sonic/data/"),
    ("assets/bm/unitree_description/", "assets/robot_description/"),
    ("1215_bones_seed_filtered", "bones_seed_smpl"),
]:
    raw = raw.replace(s, r)
raw = raw.replace(
    "assetRoot: gear_sonic/data/assets/robot_description/mjcf/",
    "assetRoot: /opt/sonic/gear_sonic/data/assets/robot_description/mjcf/",
)
cfg = omegaconf.OmegaConf.load(io.StringIO(raw))
with omegaconf.open_dict(cfg):
    cfg.experiment_dir = dest          # so exported/ lands beside the checkpoint
    cfg.headless = True
    cfg.num_envs = 1
    cfg.manager_env.commands.motion.motion_lib_cfg.motion_file = motion
    cfg.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file = "zeros"
    # the export only needs one clean reset, not a random mid-motion sample
    try:
        cfg.manager_env.commands.motion.motion_lib_cfg.start_from_first_frame = True
    except Exception:
        pass
    ev = cfg.manager_env.get("events", None)
    if ev is not None and "push_robot" in ev:
        ev.push_robot.interval_range_s = [1.0e9, 1.0e9]
omegaconf.OmegaConf.save(cfg, cfg_path)
print("patched", cfg_path)
print("experiment_dir ->", cfg.experiment_dir)
print("motion_file    ->", cfg.manager_env.commands.motion.motion_lib_cfg.motion_file)
PY

python gear_sonic/eval_agent_trl.py \
  "+checkpoint=$DEST/last.pt" \
  "+num_envs=1" \
  "+headless=true" \
  "+export_onnx_only=true"

echo "=== exported ==="
ls -la "$DEST/exported/"
python - "$DEST/exported" <<'PY'
import glob
import os
import sys

files = sorted(glob.glob(os.path.join(sys.argv[1], "*.onnx")))
assert files, "no ONNX files were produced"
for f in files:
    print(f"{os.path.getsize(f) / 1e6:8.2f} MB  {os.path.basename(f)}")
print(f"EXPORT_OK {len(files)} files")
PY
