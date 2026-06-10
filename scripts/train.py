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
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

LEROBOT_PLUGIN_ARG = "--env.discover_packages_path=motrix_il.lerobot"


def get_train_backend(supports, train_backend_arg: str | None, rllib: str):
    """Determine the training backend based on device support and user input."""
    if rllib == "rslrl":
        if train_backend_arg is not None and train_backend_arg != "torch":
            raise RuntimeError("RSLRL only supports PyTorch backend.")
        if not supports.torch:
            raise RuntimeError("RSLRL requires PyTorch, but it is not available on your device.")
        return "torch"

    if train_backend_arg is not None:
        backend = train_backend_arg
        if backend == "jax" and not supports.jax:
            raise RuntimeError("JAX is not available on your device.")
        if backend == "torch" and not supports.torch:
            raise RuntimeError("PyTorch is not available on your device.")
        return backend

    if supports.jax and supports.jax_gpu:
        return "jax"
    if supports.torch and supports.torch_gpu:
        return "torch"
    if supports.jax:
        return "jax"
    if supports.torch:
        return "torch"
    raise RuntimeError("Neither JAX nor PyTorch is available on the device.")


def add_lerobot_plugin_arg(args: list[str]) -> list[str]:
    for arg in args:
        if arg == "--env.discover_packages_path" or arg.startswith("--env.discover_packages_path="):
            return args
    return [LEROBOT_PLUGIN_ARG, *args]


def run_lerobot_train(args: list[str]) -> int:
    executable = shutil.which("lerobot-train")
    if executable is None:
        raise RuntimeError("Could not find 'lerobot-train'. Install the IL extras first.")
    command = [executable, *add_lerobot_plugin_arg(args)]
    logger.info("Running: %s", " ".join(command))
    return subprocess.run(command).returncode


def run_rl_train(args: argparse.Namespace) -> None:
    from skrl import config

    from motrix_rl import utils

    device_supports = utils.get_device_supports()
    logger.info(device_supports)

    rl_override = {}
    if args.num_envs is not None:
        rl_override["num_envs"] = args.num_envs

    if args.rand_seed:
        rl_override["runner.seed"] = None
    elif args.seed is not None:
        rl_override["runner.seed"] = args.seed

    train_backend = get_train_backend(device_supports, args.train_backend, args.rllib)

    if args.rllib == "rslrl":
        assert device_supports.torch, "PyTorch is not available on your device"
        assert train_backend == "torch", "RSLRL only supports PyTorch backend"
        from motrix_rl.rslrl.torch.train import ppo

        trainer = ppo.Trainer(args.env, args.sim_backend, cfg_override=rl_override, enable_render=args.render)
    elif train_backend == "jax":
        from motrix_rl.skrl.jax.train import ppo

        config.jax.backend = "jax"
        trainer = ppo.Trainer(args.env, args.sim_backend, cfg_override=rl_override, enable_render=args.render)
    elif train_backend == "torch":
        from motrix_rl.skrl.torch.train import ppo

        trainer = ppo.Trainer(args.env, args.sim_backend, cfg_override=rl_override, enable_render=args.render)
    else:
        raise RuntimeError(f"Unknown train backend: {train_backend}")

    trainer.train()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified MotrixLab training entry. Use --backend rl for Motrix RL, "
            "or --backend il to forward remaining arguments to lerobot-train."
        )
    )
    parser.add_argument("--backend", choices=["rl", "il"], default="rl", help="Training backend.")
    parser.add_argument("--env", default="cartpole", help="RL environment name.")
    parser.add_argument("--sim-backend", default=None, help="RL simulation backend.")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of RL envs to train.")
    parser.add_argument("--render", action="store_true", help="Render the RL environment while training.")
    parser.add_argument("--train-backend", choices=["jax", "torch"], default=None, help="RL learning backend.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--rand-seed", action="store_true", help="Generate a random RL seed.")
    parser.add_argument("--rllib", choices=["skrl", "rslrl"], default="skrl", help="RL framework.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)

    if args.backend == "il":
        return run_lerobot_train(passthrough)

    if passthrough:
        parser.error(f"Unknown RL arguments: {' '.join(passthrough)}")
    run_rl_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
