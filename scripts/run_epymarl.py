#!/usr/bin/env python3
"""Run small EPyMARL examples for STAT, LBF, or RWARE.

The defaults are intentionally tiny smoke-test settings. Increase --steps,
--test-nepisode, and --seed or launch multiple seeds for paper-scale runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EPYMARL_ROOT = REPO_ROOT / "third_party" / "epymarl"
EPYMARL_SRC = EPYMARL_ROOT / "src"
STAT_SRC = REPO_ROOT / "src"

METHODS = ("iql", "vdn", "qmix", "mappo", "ippo", "maa2c")
ENVS = ("stat", "lbf", "rware")

DEFAULT_ENV_IDS = {
    "lbf": "lbforaging:Foraging-8x8-2p-2f-coop-v3",
    "rware": "rware:rware-tiny-2ag-v2",
}


def build_env() -> dict[str, str]:
    """Expose the released STAT package and vendored EPyMARL tree to Python."""
    env = os.environ.copy()
    paths = [str(STAT_SRC), str(EPYMARL_SRC)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="mappo")
    parser.add_argument("--env", choices=ENVS, default="stat")
    parser.add_argument("--env-id", default=None, help="Gymnasium ID for LBF/RWARE.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--test-nepisode", type=int, default=2)
    parser.add_argument("--test-interval", type=int, default=None)
    parser.add_argument("--results-dir", default="results/smoke/epymarl")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--common-reward", action="store_true", help="Use EPyMARL common team reward.")
    parser.add_argument("--reward-scalarisation", default="sum", choices=("sum", "mean"))
    parser.add_argument("--stat-agents", type=int, default=3)
    parser.add_argument("--stat-tasks", type=int, default=6)
    parser.add_argument("--stat-width", type=int, default=5)
    parser.add_argument("--stat-height", type=int, default=3)
    parser.add_argument("--stat-bins", type=int, default=5)
    parser.add_argument("--episode-limit", type=int, default=50)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    """Translate the compact wrapper arguments into EPyMARL Sacred overrides."""
    main_py = EPYMARL_SRC / "main.py"
    test_interval = args.test_interval if args.test_interval is not None else max(1, args.steps // 2)
    cmd = [
        sys.executable,
        str(main_py),
        f"--config={args.method}",
        f"--env-config={env_config(args.env)}",
        "with",
        f"use_cuda={str(args.use_cuda)}",
        f"seed={args.seed}",
        f"t_max={args.steps}",
        f"test_nepisode={args.test_nepisode}",
        f"test_interval={test_interval}",
        "save_model=False",
        "use_tensorboard=False",
        "buffer_cpu_only=True",
        f"local_results_path={REPO_ROOT / args.results_dir}",
        f"name={args.method.upper()}_{args.env.upper()}_SMOKE",
    ]

    if args.env == "stat":
        cmd.extend([
            f"env_args.n_agents={args.stat_agents}",
            f"env_args.n_tasks={args.stat_tasks}",
            f"env_args.width={args.stat_width}",
            f"env_args.height={args.stat_height}",
            f"env_args.num_bins={args.stat_bins}",
            f"env_args.episode_limit={args.episode_limit}",
        ])
    else:
        env_id = args.env_id or DEFAULT_ENV_IDS[args.env]
        cmd.extend([
            f"env_args.key={env_id}",
            f"env_args.time_limit={args.episode_limit}",
            f"common_reward={str(args.common_reward)}",
            f"reward_scalarisation={args.reward_scalarisation}",
        ])

    return cmd


def env_config(env_name: str) -> str:
    if env_name == "stat":
        return "stat"
    if env_name == "lbf":
        return "lbf_metrics"
    if env_name == "rware":
        return "rware_metrics"
    raise ValueError(env_name)


def main() -> None:
    args = parse_args()
    cmd = build_command(args)
    subprocess.run(cmd, cwd=EPYMARL_SRC, env=build_env(), check=True)


if __name__ == "__main__":
    main()
