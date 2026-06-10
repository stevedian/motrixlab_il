# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import subprocess

logger = logging.getLogger(__name__)

LEROBOT_PLUGIN_ARG = "--env.discover_packages_path=motrix_il.lerobot"


def get_inference_backend(policy_path: Path | str, rllib: str):
    """Determine the backend from RL framework and policy file extension."""
    if rllib == "rslrl":
        return "torch"

    suffix = policy_path.suffix if isinstance(policy_path, Path) else Path(policy_path).suffix
    if suffix == ".pt":
        return "torch"
    if suffix == ".pickle":
        return "jax"
    raise RuntimeError(f"Unknown policy format: {policy_path}")


def discover_rllib(env_name: str) -> tuple[str, Path]:
    """Discover the RL framework and best policy from the most recent training run."""
    base_dir = Path(f"runs/{env_name}")
    if not base_dir.exists():
        raise FileNotFoundError(f"No training results found for environment '{env_name}' in {base_dir}")

    frameworks = []
    for framework in ["skrl", "rslrl"]:
        framework_dir = base_dir / framework
        if framework_dir.exists() and framework_dir.is_dir():
            training_runs = [d for d in framework_dir.iterdir() if d.is_dir()]
            if training_runs:
                latest_run = max(training_runs, key=lambda x: x.stat().st_mtime)
                frameworks.append((framework, latest_run.stat().st_mtime, latest_run))

    if not frameworks:
        raise FileNotFoundError(f"No training runs found for environment '{env_name}' in {base_dir}")

    latest_framework, _, latest_run_dir = max(frameworks, key=lambda x: x[1])

    if latest_framework == "rslrl":
        model_files = list(latest_run_dir.glob("model_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No policy files found in {latest_run_dir}")

        def extract_iteration(filename):
            parts = Path(filename).stem.split("_")
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return 0
            return 0

        best_policy = max(model_files, key=lambda f: (f.stat().st_mtime, extract_iteration(f)))
    else:
        checkpoints_dir = latest_run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No checkpoints directory found in {latest_run_dir}")

        best_files = list(checkpoints_dir.glob("best_agent.*"))
        if best_files:
            best_policy = best_files[0]
        else:
            checkpoint_files = list(checkpoints_dir.glob("agent_*.pt")) + list(checkpoints_dir.glob("agent_*.pickle"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No policy files found in {checkpoints_dir}")

            def extract_timestep(filename):
                parts = Path(filename).stem.split("_")
                if len(parts) >= 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        return 0
                return 0

            best_policy = max(checkpoint_files, key=extract_timestep)

    return latest_framework, best_policy


def add_lerobot_plugin_arg(args: list[str]) -> list[str]:
    for arg in args:
        if arg == "--env.discover_packages_path" or arg.startswith("--env.discover_packages_path="):
            return args
    return [LEROBOT_PLUGIN_ARG, *args]


def run_lerobot_eval(args: list[str]) -> int:
    executable = shutil.which("lerobot-eval")
    if executable is None:
        raise RuntimeError("Could not find 'lerobot-eval'. Install the IL extras first.")
    command = [executable, *add_lerobot_plugin_arg(args)]
    logger.info("Running: %s", " ".join(command))
    return subprocess.run(command).returncode


def run_rl_play(args: argparse.Namespace) -> None:
    from skrl import config

    from motrix_rl import utils

    device_supports = utils.get_device_supports()
    logger.info(device_supports)

    rl_override = {}
    if args.num_envs is not None:
        rl_override["play_num_envs"] = args.num_envs

    if args.rand_seed:
        rl_override["runner.seed"] = None
    elif args.seed is not None:
        rl_override["runner.seed"] = args.seed

    if args.policy:
        if not args.rllib:
            raise RuntimeError("--policy requires --rllib for RL playback.")
        rllib = args.rllib
        policy_path = args.policy
        logger.info("Using specified RL framework: %s", rllib)
        logger.info("Using specified policy: %s", policy_path)
    else:
        rllib, policy_path = discover_rllib(args.env)
        logger.info("Auto-discovered RL framework: %s", rllib)
        logger.info("Auto-discovered best policy: %s", policy_path)

    backend = get_inference_backend(policy_path, rllib)

    if rllib == "rslrl":
        assert device_supports.torch, "PyTorch is not available on your device"
        from motrix_rl.rslrl.torch.train import ppo

        config.torch.backend = "torch"
        trainer = ppo.Trainer(args.env, args.sim_backend, cfg_override=rl_override, enable_render=True)
        trainer.play(policy_path)
    elif backend == "jax":
        assert device_supports.jax, "JAX is not available on your device"
        from motrix_rl.skrl.jax.train import ppo

        config.jax.backend = "jax"
        trainer = ppo.Trainer(args.env, args.sim_backend, cfg_override=rl_override, enable_render=True)
        trainer.play(policy_path)
    elif backend == "torch":
        assert device_supports.torch, "PyTorch is not available on your device"
        from motrix_rl.skrl.torch.train import ppo

        config.torch.backend = "torch"
        trainer = ppo.Trainer(args.env, args.sim_backend, cfg_override=rl_override, enable_render=True)
        trainer.play(policy_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified MotrixLab playback/evaluation entry. Use --backend rl for Motrix RL, "
            "or --backend il to forward remaining arguments to lerobot-eval."
        )
    )
    parser.add_argument("--backend", choices=["rl", "il"], default="rl", help="Playback backend.")
    parser.add_argument("--env", default="cartpole", help="RL environment name.")
    parser.add_argument("--sim-backend", default=None, help="RL simulation backend.")
    parser.add_argument("--policy", default=None, help="RL policy checkpoint path.")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of RL envs to play.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--rand-seed", action="store_true", help="Generate a random RL seed.")
    parser.add_argument("--rllib", choices=["skrl", "rslrl"], default=None, help="RL framework.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)

    if args.backend == "il":
        return run_lerobot_eval(passthrough)

    if passthrough:
        parser.error(f"Unknown RL arguments: {' '.join(passthrough)}")
    run_rl_play(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
