"""Headless free-dynamics rollout of a SONIC checkpoint -> fall report.

This is the render pipeline of deploy/render_sim_video.py with the RTX camera
removed, so it runs on a plain CUDA container (no Vulkan). It answers the one
question that matters for a demo video: does the policy stay on its feet for the
whole motion, or does it diverge and fall?

All terminations except time_out are disabled here on purpose. With them on, a
diverging policy is reset onto the reference and the roll-out looks fine - the
fall only reappears as a teleport in the render.

Usage:
  python verify_rollout.py --ckpt <last.pt> --motion <motion.pkl> --out <tracks.npz>
"""
import argparse
import datetime as dt
import io
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
    p.add_argument("--motion", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=444)
    a = p.parse_args()

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
    raw = raw.replace(
        "assetRoot: gear_sonic/data/assets/robot_description/mjcf/",
        "assetRoot: /opt/sonic/gear_sonic/data/assets/robot_description/mjcf/",
    )
    cfg = omegaconf.OmegaConf.load(io.StringIO(raw))
    over = omegaconf.OmegaConf.create({
        "headless": True,
        "num_envs": 1,
        "use_wandb": False,
        "manager_env.commands.motion.motion_lib_cfg.motion_file": a.motion,
        "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file": "zeros",
    })
    cfg = omegaconf.OmegaConf.merge(cfg, over)
    with omegaconf.open_dict(cfg):
        if "eval_overrides" in cfg:
            del cfg["eval_overrides"]
        cfg.manager_env.commands.motion.motion_lib_cfg.motion_file = a.motion
        cfg.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file = "zeros"
        ev = cfg.manager_env.get("events", None)
        if ev is not None and "push_robot" in ev:
            ev.push_robot.interval_range_s = [1.0e9, 1.0e9]
    with omegaconf.open_dict(cfg.manager_env):
        for name, term in cfg.manager_env.terminations.items():
            if not isinstance(term, omegaconf.DictConfig):
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

    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    ap_args, _ = ap.parse_known_args()
    ap_args.headless = True
    ap_args.enable_cameras = False
    ap_args.num_envs = 1
    ap_args.seed = 0
    ap_args.multi_gpu = False
    ap_args.distributed = False
    ap_args.device = "cuda:0"
    ap_args.env_spacing = cfg.manager_env.config.env_spacing
    ap_args.output_dir = cfg.output_dir
    ap_args.kit_args = "--/log/level=error --/log/fileLogLevel=error"
    AppLauncher(ap_args)

    import accelerate
    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
    from gear_sonic.trl.utils.common import custom_instantiate
    from gear_sonic.utils.common import seeding

    accelerator = accelerate.Accelerator(
        kwargs_handlers=[accelerate.InitProcessGroupKwargs(timeout=dt.timedelta(seconds=300))]
    )
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

    from gear_sonic.utils import obs_utils

    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key not in ["policy", "critic"]:
            d, n, tot = obs_utils.get_group_term_obs_shape(example_obs, key)
            env.config["obs"]["group_obs_dims"][key] = d
            env.config["obs"]["group_obs_names"][key] = n
            env.config["obs"]["obs_dims"][key] = tot
            env.config["robot"]["algo_obs_dim_dict"][key] = tot
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = \
        env.env.observation_space["policy"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = \
        env.env.observation_space["critic"].shape[-1]
    meta_action_dim = env.config.get("meta_action_dim", None)
    env.config["robot"]["actions_dim"] = (
        meta_action_dim if meta_action_dim is not None and meta_action_dim > 0
        else env.env.action_space.shape[-1]
    )

    from gear_sonic.trl.trainer import ppo_trainer

    module_dim_dict = getattr(cfg.algo.config, "module_dim", {})
    policy = custom_instantiate(
        cfg.algo.config.actor, env_config=env.config, algo_config=cfg.algo.config,
        module_dim_dict=module_dim_dict, backbone_kwargs={}, _resolve=False,
    ).to(device)
    value_model = custom_instantiate(
        cfg.algo.config.critic, env_config=env.config, algo_config=cfg.algo.config,
        module_dim_dict=module_dim_dict, backbone_kwargs={}, _resolve=False,
    ).to(device)
    model = ppo_trainer.PolicyAndValueWrapper(policy, value_model)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    model.policy.load_state_dict(ck.get("actor_model_state_dict", ck.get("policy_state_dict", {})))
    print(f"LOADED {a.ckpt}", flush=True)

    model.policy.init_rollout()
    obs_dict = env.reset_all()
    for k in obs_dict:
        obs_dict[k] = obs_dict[k].to(device)
    mc = env.motion_command
    for attr in ("motion_start_time_steps", "time_steps"):
        t = getattr(mc, attr, None)
        if t is not None and hasattr(t, "zero_"):
            t.zero_()

    rb = env.env.scene["robot"]
    bidx = rb.find_bodies(env.motion_command.cfg.body_names, preserve_order=True)[0]

    pred_all, gt_all = [], []
    for i in range(a.max_steps):
        actions = model.policy.rollout(obs_dict=obs_dict)
        act = {"actions": model.policy.action_mean.detach(), "obs_dict": actions["obs_dict"]}
        obs_dict, _, dones, _ = env.step(act)
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].to(device)
        pred_all.append(rb.data.body_pos_w.detach().cpu().numpy()[0, bidx])
        gt_all.append(env.motion_command.body_pos_w.detach().cpu().numpy()[0])
        if bool(dones.any()):
            print(f"  episode ended at step {i}", flush=True)
            break
    pred = np.stack(pred_all)
    gt = np.stack(gt_all)
    np.savez(a.out, pred_pos_all=pred, gt_pos_all=gt)

    # ---- fall report -------------------------------------------------------
    z = pred[:, 0, 2]
    ref_z = gt[:, 0, 2]
    step = np.linalg.norm(np.diff(pred, axis=0), axis=-1).max(axis=1) * 1000.0
    below = np.where(z < 0.45)[0]
    print("=" * 60)
    print(f"STEPS {len(z)}  (motion needs {a.max_steps})")
    print(f"root height  : min {z.min():.3f}  end {z[-1]:.3f}  (reference {ref_z.min():.3f}-{ref_z.max():.3f})")
    print(f"max frame step: {step.max():.0f} mm  median {np.median(step):.0f} mm")
    if len(below):
        print(f"FALL ONSET   : frame {int(below[0])}  (t={below[0] / 50.0:.2f}s)  <-- FAILED")
        print("VERDICT: FELL")
    elif len(z) < a.max_steps - 2:
        print("VERDICT: ENDED EARLY (episode terminated before the motion finished)")
    else:
        print("VERDICT: COMPLETED_UPRIGHT")
    print("=" * 60)
    print(f"SAVED {a.out}")


if __name__ == "__main__":
    main()
