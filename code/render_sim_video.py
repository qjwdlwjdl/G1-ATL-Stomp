"""Render a TRUE Isaac Sim simulation video of the G1 policy (like the UltimateBots BEFORE/AFTER mp4s).

Runs the free-dynamics policy rollout (same as rollout2.py) while rendering every
physics step (50 Hz sim -> 50 fps video, real-time speed) through an offscreen RTX
camera. All terminations (except time_out) are disabled so the episode never
resets mid-motion - a reset would teleport the robot to a new motion frame.

REQUIRES: export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json  (excludes llvmpipe,
otherwise omni.gpu.foundation picks the software device and RTX init fails).

Usage:
  python render_sim_video.py --ckpt <path> --out <mp4> [--num-envs 1] [--gpu 0]
Outputs: <out>.mp4 + <out>.npz (pred/gt tracks, same format as rollout2.py)
"""
import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, "/opt/sonic")
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import numpy as np
import omegaconf
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--motion", default="/opt/sonic/data/atl_stomp_fixed.pkl")
    p.add_argument("--gpu", type=int, default=0, help="GPU index for render+physics+torch")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=296,
                   help="sim steps; ATL Stomp v4 motion is 178 frames @30fps "
                        "= 296 physics steps @50Hz. Do not go past the motion "
                        "end (time_out resets the episode).")
    a = p.parse_args()

    import io
    # The MJCF robot asset root is a relative path in config.yaml; Isaac Lab
    # resolves it from $CWD, which is NOT /opt/sonic when launched via nohup.
    # Chdir to /opt/sonic and force the asset root to an absolute path so the
    # G1 MJCF import succeeds (else the pelvis.tmp.usd layer is never written
    # and USD crashes with a NULL TfRefPtr<UsdStage>).
    if os.path.isdir("/opt/sonic"):
        os.chdir("/opt/sonic")
    ckpt_dir = os.path.dirname(a.ckpt)
    raw = open(os.path.join(ckpt_dir, "config.yaml"), encoding="utf-8").read()
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
    # force the MJCF asset root + motion file to absolute paths (cwd-safe)
    raw = raw.replace(
        "assetRoot: gear_sonic/data/assets/robot_description/mjcf/",
        "assetRoot: /opt/sonic/gear_sonic/data/assets/robot_description/mjcf/",
    )
    cfg = omegaconf.OmegaConf.load(io.StringIO(raw))
    over = omegaconf.OmegaConf.create(
        {
            "headless": True,
            "num_envs": a.num_envs,
            "use_wandb": False,
            "manager_env.commands.motion.motion_lib_cfg.motion_file": a.motion,
            "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file": "zeros",
        }
    )
    cfg = omegaconf.OmegaConf.merge(cfg, over)
    with omegaconf.open_dict(cfg):
        if "eval_overrides" in cfg:
            del cfg["eval_overrides"]
        cfg.manager_env.commands.motion.motion_lib_cfg.motion_file = a.motion
        cfg.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file = "zeros"
        # disable the push_robot domain-randomization event: it kicks the robot
        # every 4-6 s during training, which shows up as a sudden jump in a demo video
        ev = cfg.manager_env.get("events", None)
        if ev is not None and "push_robot" in ev:
            ev.push_robot.interval_range_s = [1.0e9, 1.0e9]
    # force the motion to start at frame 0: reset samples a random in-motion
    # start index; past the motion end it wraps and looks like a "reset"
    with omegaconf.open_dict(cfg.manager_env):
        # DISABLE every termination except time_out. A terminated episode
        # makes env.step() reset the episode (motion jumps to a new random
        # frame) and the robot teleports - that is the "pose jump / reset"
        # seen mid-video. With terminations off the policy rolls the whole
        # motion through in one continuous trajectory.
        terms = cfg.manager_env.terminations
        for name, term in terms.items():
            if not isinstance(term, omegaconf.DictConfig):  # e.g. _target_: str
                continue
            if "time_out" in name.lower():
                continue
            with omegaconf.open_dict(term):
                params = term.get("params") if "params" in term else None
                if params is not None:
                    with omegaconf.open_dict(params):
                        for k in ("threshold", "down_threshold"):
                            if k in params:
                                params[k] = 1.0e6
                        if "threshold_adaptive" in params:
                            params["threshold_adaptive"] = False
                if "disabled" in term:
                    term["disabled"] = True
    print("terminations disabled (only time_out remains)", flush=True)
    print("motion_file:", cfg.manager_env.commands.motion.motion_lib_cfg.motion_file, flush=True)

    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    ap_args, _ = ap.parse_known_args()
    ap_args.headless = True
    ap_args.enable_cameras = True  # init the RTX render stack
    ap_args.num_envs = a.num_envs
    ap_args.seed = 0
    ap_args.multi_gpu = False
    ap_args.distributed = False
    ap_args.device = f"cuda:{a.gpu}"
    ap_args.env_spacing = cfg.manager_env.config.env_spacing
    ap_args.output_dir = cfg.output_dir
    ap_args.kit_args = (
        f"--/log/level=error --/log/fileLogLevel=error "
        f"--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false "
        f"--/physics/cudaDevice={a.gpu}"
    )
    app_launcher = AppLauncher(ap_args)
    sim_app = app_launcher.app

    import accelerate
    import cv2
    import hydra
    import omni.replicator.core as rep
    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
    from gear_sonic.trl.utils.common import custom_instantiate
    from gear_sonic.utils.common import seeding

    accelerator_kwargs = accelerate.InitProcessGroupKwargs(timeout=dt.timedelta(seconds=300))
    accelerator = accelerate.Accelerator(kwargs_handlers=[accelerator_kwargs])
    device = str(accelerator.device)
    torch.cuda.set_device(accelerator.local_process_index)
    seeding(cfg.seed)

    from isaaclab.envs import ManagerBasedRLEnv

    env_instance_cfg = custom_instantiate(cfg.manager_env)
    env_instance_cfg.seed = cfg.seed
    env_instance_cfg.sim.device = device
    env_instance_cfg.config["headless"] = True
    env = ManagerBasedRLEnv(cfg=env_instance_cfg, render_mode=None)
    env = ManagerEnvWrapper(env, env_instance_cfg.config)

    # --- scene dressing for the render: floor + dome light ---
    import time

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    while stage is None:  # parallel starts can race the stage open
        time.sleep(1.0)
        stage = omni.usd.get_context().get_stage()
    from pxr import Gf, UsdGeom, UsdLux

    plane = UsdGeom.Mesh.Define(stage, "/World/RenderFloor")
    size = 6.0
    pts = [(-size, -size, 0), (size, -size, 0), (size, size, 0), (-size, size, 0)]
    plane.CreatePointsAttr([Gf.Vec3f(*q) for q in pts])
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.GetIntensityAttr().Set(1500.0)

    # --- camera: fixed 3/4 view, re-aimed at the robot root each frame ---
    cam_prim = UsdGeom.Camera.Define(stage, "/World/SimCam")
    cam_prim.GetFocalLengthAttr().Set(24.0)
    xform = UsdGeom.Xformable(cam_prim)
    rp = rep.create.render_product(str(cam_prim.GetPath()), (a.width, a.height))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach(rp)

    # --- obs dims (mirror eval) ---
    from gear_sonic.utils import obs_utils

    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key not in ["policy", "critic"]:
            group_obs_dims, group_obs_names, group_obs_total_dim = (
                obs_utils.get_group_term_obs_shape(example_obs, key)
            )
            env.config["obs"]["group_obs_dims"][key] = group_obs_dims
            env.config["obs"]["group_obs_names"][key] = group_obs_names
            env.config["obs"]["obs_dims"][key] = group_obs_total_dim
            env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]
    meta_action_dim = env.config.get("meta_action_dim", None)
    env.config["robot"]["actions_dim"] = (
        meta_action_dim
        if meta_action_dim is not None and meta_action_dim > 0
        else env.env.action_space.shape[-1]
    )

    from gear_sonic.trl.trainer import ppo_trainer

    policy_backbone_kwargs = {}
    module_dim_dict = getattr(cfg.algo.config, "module_dim", {})
    policy = custom_instantiate(
        cfg.algo.config.actor,
        env_config=env.config,
        algo_config=cfg.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs=policy_backbone_kwargs,
        _resolve=False,
    ).to(device)
    value_model = custom_instantiate(
        cfg.algo.config.critic,
        env_config=env.config,
        algo_config=cfg.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs=policy_backbone_kwargs,
        _resolve=False,
    ).to(device)

    import easydict

    args = easydict.EasyDict()
    args.is_main_process = accelerator.is_main_process
    args.global_rank = accelerator.process_index
    args.world_size = accelerator.num_processes
    state = easydict.EasyDict()

    model = ppo_trainer.PolicyAndValueWrapper(policy, value_model)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    sd = ck.get("actor_model_state_dict", ck.get("policy_state_dict", {}))
    model.policy.load_state_dict(sd)
    print(f"LOADED {a.ckpt}", flush=True)

    # --- rollout + render ---
    model.policy.init_rollout()
    obs_dict = env.reset_all()
    for k in obs_dict:
        obs_dict[k] = obs_dict[k].to(device)

    # force the motion to start at frame 0: reset samples a random in-motion
    # start index; past the motion end it wraps and looks like a "reset"
    mc = env.motion_command
    for attr in ("motion_start_time_steps", "time_steps"):
        if hasattr(mc, attr):
            t = getattr(mc, attr)
            if hasattr(t, "zero_"):
                t.zero_()
                print(f"zeroed motion_command.{attr}", flush=True)

    rb = env.env.scene["robot"]
    body_names = env.motion_command.cfg.body_names
    bidx = rb.find_bodies(body_names, preserve_order=True)[0]

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.width, a.height))
    assert vw.isOpened(), "VideoWriter failed"

    pred_pos_all = []
    gt_pos_all = []
    max_steps = a.max_steps
    frame_every = 1  # render every physics step -> 50 fps video, no dropped motion
    for i in range(max_steps):
        actions = model.policy.rollout(obs_dict=obs_dict)
        act = {"actions": model.policy.action_mean.detach(), "obs_dict": actions["obs_dict"]}
        obs_dict, _, dones, _ = env.step(act)
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].to(device)

        pred = rb.data.body_pos_w.detach().cpu().numpy()[:, bidx]  # (E,14,3) simulated
        ref = env.motion_command.body_pos_w.detach().cpu().numpy()  # (E,14,3) reference
        pred_pos_all.append(pred[0])
        gt_pos_all.append(ref[0])

        if bool(dones.any()):
            # episode ended this step (time_out at motion end): env.step
            # already reset the episode inside, so render nothing past it -
            # the frames written so far cover the whole motion.
            print(f"  episode ended at step {i}, stopping", flush=True)
            break

        if i % frame_every == 0:
            # keep the fixed camera pointed at the robot root
            root = pred[0, 0]
            look = root + np.array([0.0, 0.0, 0.05])
            eye = look + np.array([2.2, -2.2, 1.35])
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*eye))

            fwd = look - eye
            fwd = fwd / np.linalg.norm(fwd)
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            up2 = np.cross(right, fwd)
            # camera axes in world: +X right, +Y up, +Z backward; mat->quat
            m = np.stack([right, up2, -fwd], axis=1)  # columns
            tr = m[0, 0] + m[1, 1] + m[2, 2]
            if tr > 0:
                s = np.sqrt(tr + 1.0) * 2
                qw, qx = 0.25 * s, (m[2, 1] - m[1, 2]) / s
                qy, qz = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
            elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
                qw, qx = (m[2, 1] - m[1, 2]) / s, 0.25 * s
                qy, qz = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
                qw, qx = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
                qy, qz = 0.25 * s, (m[1, 2] + m[2, 1]) / s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
                qw, qx = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
                qy, qz = (m[1, 2] + m[2, 1]) / s, 0.25 * s
            xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(float(qw), float(qx), float(qy), float(qz)))
            # render WITHOUT stepping physics (orchestrator.step blocks in this env)
            for _ in range(6):
                env.env.sim.render()
            img = np.asarray(annot.get_data(), dtype=np.uint8)[..., :3]
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if i % 40 == 0:
                print(f"  frame {i} rendered", flush=True)

    vw.release()
    pred_pos_all = np.stack(pred_pos_all)
    gt_pos_all = np.stack(gt_pos_all)
    np.savez(a.out + ".tracks.npz", pred_pos_all=pred_pos_all, gt_pos_all=gt_pos_all)
    print(f"SAVED {a.out} + tracks ({pred_pos_all.shape[0]} sim steps)", flush=True)
    sim_app.close()


if __name__ == "__main__":
    main()
