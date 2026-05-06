#!/usr/bin/env python3
"""Run a tiny configurable smoke test for any released method."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
PYMARL_SRC = REPO_ROOT / "third_party" / "pymarl" / "src"
for import_path in (SRC_DIR, PYMARL_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

METHODS = ("stat", "dqn", "fdqn", "iql", "vdn", "qmix", "qtran", "coma")
PYMARL_METHODS = {"iql", "vdn", "qmix", "qtran", "coma"}
DEFAULT_MAX_DQN_JOINT_ACTIONS = 250_000


def _env(extra: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(SRC_DIR), str(PYMARL_SRC)]
    if extra:
        parts.append(extra)
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env or _env(), check=True)


def _joint_action_count(agents: int, tasks: int) -> int:
    return (3 + tasks) ** agents


def _print_problem_size(args: argparse.Namespace) -> None:
    per_agent_actions = 3 + args.tasks
    joint_actions = _joint_action_count(args.agents, args.tasks)
    cells = args.width * args.height
    density = args.tasks / cells
    print(
        "Problem size: "
        f"per_agent_actions={per_agent_actions} "
        f"dqn_joint_actions={joint_actions:,} "
        f"grid_cells={cells} task_density={density:.3f}",
        flush=True,
    )


def _validate_and_warn(args: argparse.Namespace, selected: tuple[str, ...]) -> None:
    cells = args.width * args.height
    density = args.tasks / cells
    per_agent_actions = 3 + args.tasks
    joint_actions = _joint_action_count(args.agents, args.tasks)

    if args.agents > args.tasks:
        print(
            "[stress-test warning] agents > tasks. This is allowed, but many agents may become idle "
            "or redundant, so smoke-test rewards may be odd.",
            flush=True,
        )

    if args.tasks > cells:
        print(
            "[stress-test warning] tasks > grid cells. This is allowed as a stress test, but task "
            "placement/dynamics may be less interpretable.",
            flush=True,
        )
    elif density > 0.75:
        print(
            "[stress-test warning] high task density. This should run, but it may not be a clean "
            "representative benchmark setting.",
            flush=True,
        )

    if any(method in {"fdqn", *PYMARL_METHODS} for method in selected) and per_agent_actions > args.max_per_agent_actions:
        print(
            "[stress-test warning] large per-agent action space: "
            f"3 + tasks = {per_agent_actions}, above the smoke-test guideline of "
            f"{args.max_per_agent_actions}. This is allowed, but may be slower or unstable.",
            flush=True,
        )

    if "dqn" in selected and joint_actions > args.max_dqn_joint_actions and not args.allow_large_dqn:
        raise SystemExit(
            "DQN joint-action space is too large for the default smoke-test guard: "
            f"(3 + tasks)^agents = {joint_actions:,}, limit={args.max_dqn_joint_actions:,}. "
            "Use --allow_large_dqn to run this intentionally as a stress test."
        )

    if "dqn" in selected and joint_actions > args.max_dqn_joint_actions:
        print(
            "[stress-test warning] large DQN joint-action space. This run may be slow or memory-heavy, "
            "and a smoke-test reward is not meaningful.",
            flush=True,
        )


def smoke_stat(args: argparse.Namespace) -> None:
    from stat_env import STATConfig, make_stat_env

    rng = np.random.default_rng(args.seed)
    env = make_stat_env(STATConfig(
        seed=args.seed,
        agents=args.agents,
        tasks=args.tasks,
        width=args.width,
        height=args.height,
        num_bins=args.num_bins,
        episode_limit=args.episode_limit,
    ))

    for episode in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        for _step in range(args.episode_limit):
            action = []
            for mask in obs["agent_masks"]:
                valid = np.flatnonzero(mask)
                if len(valid) == 0:
                    raise RuntimeError("STAT returned an agent with no valid actions.")
                action.append(int(rng.choice(valid)))

            obs, reward, done, _truncated, info = env.step(action)
            total_reward += float(reward)

            for key in ["forced_idle", "num_conflicts", "unique_tasks_assigned", "J_upper"]:
                if key not in info:
                    raise RuntimeError(f"STAT step info is missing metric: {key}")

            if done:
                break

        print(f"[stat] episode={episode} reward={total_reward:.3f}", flush=True)

    print("STAT smoke test passed.", flush=True)


def smoke_dqn_like(method: str, args: argparse.Namespace) -> None:
    script = SRC_DIR / method / f"train_{method}.py"
    output_dir = REPO_ROOT / "results" / "smoke" / method

    cmd = [
        sys.executable,
        str(script),
        "--agents",
        str(args.agents),
        "--tasks",
        str(args.tasks),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--num_bins",
        str(args.num_bins),
        "--seed",
        str(args.seed),
        "--total_steps",
        str(args.steps),
        "--batch_size",
        str(args.batch_size),
        "--out_dir",
        str(output_dir),
        "--save_every_steps",
        str(max(args.steps + 1, 2)),
        "--eval_max_steps",
        str(args.episode_limit),
        "--epsilon_decay_steps",
        str(max(1, args.steps)),
    ]

    if method == "fdqn":
        cmd.extend(["--hidden_dim", str(args.hidden_dim)])

    try:
        _run(cmd)
    finally:
        if args.clean and output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Removed smoke output: {output_dir}", flush=True)


def smoke_pymarl(method: str, args: argparse.Namespace) -> None:
    main_py = PYMARL_SRC / "main.py"
    results_dir = REPO_ROOT / "results" / "smoke" / method

    cmd = [
        sys.executable,
        str(main_py),
        f"--config={method}",
        "--env-config=stat",
        "with",
        "use_cuda=False",
        f"local_results_path={results_dir}",
        f"name={method.upper()}_SMOKE",
        "label=smoke_test",
        f"env_args.agents={args.agents}",
        f"env_args.tasks={args.tasks}",
        f"env_args.width={args.width}",
        f"env_args.height={args.height}",
        f"env_args.num_bins={args.num_bins}",
        f"env_args.episode_limit={args.episode_limit}",
        f"seed={args.seed}",
        f"t_max={args.steps}",
        f"test_interval={max(1, args.steps // 2)}",
        "test_nepisode=1",
        "runner_log_interval=100",
        "save_model=False",
        "use_tensorboard=False",
        "buffer_cpu_only=True",
    ]

    _run(cmd)
    if args.clean and results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"Removed smoke output: {results_dir}", flush=True)
    sacred_dir = REPO_ROOT / "results" / "sacred"
    if args.clean and sacred_dir.exists():
        shutil.rmtree(sacred_dir)
        print(f"Removed smoke output: {sacred_dir}", flush=True)


def run_method(method: str, args: argparse.Namespace) -> None:
    if method == "stat":
        smoke_stat(args)
    elif method in {"dqn", "fdqn"}:
        smoke_dqn_like(method, args)
    elif method in PYMARL_METHODS:
        smoke_pymarl(method, args)
    else:
        raise ValueError(f"Unknown method: {method}")


def _load_config_defaults(argv: list[str]) -> dict[str, object]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    known, _unknown = pre_parser.parse_known_args(argv)
    if known.config is None:
        return {}

    config_path = Path(known.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise SystemExit(f"Smoke config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Smoke config must be a YAML mapping: {config_path}")
    return data


def _build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny configurable smoke test for STAT or any released MARL method."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config with smoke-test defaults.")
    parser.add_argument(
        "--method",
        choices=METHODS + ("all",),
        default=defaults.get("method", "stat"),
        help="Method to smoke test. Use 'all' to run every method.",
    )
    parser.add_argument("--agents", "--num-agents", type=int, default=defaults.get("agents", 3), help="Number of agents.")
    parser.add_argument("--tasks", "--num-tasks", type=int, default=defaults.get("tasks", 6), help="Number of tasks.")
    parser.add_argument("--width", type=int, default=defaults.get("width", 5), help="Grid width.")
    parser.add_argument("--height", type=int, default=defaults.get("height", 3), help="Grid height.")
    parser.add_argument("--num_bins", type=int, default=defaults.get("num_bins", 5), help="Number of distance bins.")
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    parser.add_argument("--steps", type=int, default=defaults.get("steps", 50), help="Tiny training-step budget for method smoke tests.")
    parser.add_argument("--episodes", type=int, default=defaults.get("episodes", 1), help="Number of episodes for the STAT-only smoke test.")
    parser.add_argument("--episode_limit", type=int, default=defaults.get("episode_limit", 50), help="Maximum steps per episode.")
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 8), help="DQN/FDQN smoke-test replay batch size.")
    parser.add_argument("--hidden_dim", type=int, default=defaults.get("hidden_dim", 16), help="FDQN hidden dimension for smoke tests.")
    parser.add_argument(
        "--max_dqn_joint_actions",
        type=int,
        default=defaults.get("max_dqn_joint_actions", DEFAULT_MAX_DQN_JOINT_ACTIONS),
        help="Default guardrail for DQN's enumerated joint-action space.",
    )
    parser.add_argument(
        "--max_per_agent_actions",
        type=int,
        default=defaults.get("max_per_agent_actions", 128),
        help="Guideline threshold for FDQN/PyMARL per-agent action spaces. Exceeding it warns but still runs.",
    )
    parser.add_argument(
        "--allow_large_dqn",
        action="store_true",
        default=bool(defaults.get("allow_large_dqn", False)),
        help="Allow DQN smoke tests above --max_dqn_joint_actions.",
    )
    parser.add_argument(
        "--keep_outputs",
        action="store_true",
        default=bool(defaults.get("keep_outputs", False)),
        help="Keep generated smoke outputs.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    defaults = _load_config_defaults(sys.argv[1:])
    valid_keys = {
        "method",
        "agents",
        "tasks",
        "width",
        "height",
        "num_bins",
        "seed",
        "steps",
        "episodes",
        "episode_limit",
        "batch_size",
        "hidden_dim",
        "max_dqn_joint_actions",
        "max_per_agent_actions",
        "allow_large_dqn",
        "keep_outputs",
    }
    unknown_keys = sorted(set(defaults) - valid_keys)
    if unknown_keys:
        raise SystemExit(f"Unknown smoke config key(s): {', '.join(unknown_keys)}")

    parser = _build_parser(defaults)
    args = parser.parse_args()

    for name in ["agents", "tasks", "width", "height", "num_bins", "steps", "episodes", "episode_limit"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive.")

    args.clean = not args.keep_outputs
    return args


def main() -> None:
    args = parse_args()
    selected = METHODS if args.method == "all" else (args.method,)
    _validate_and_warn(args, selected)

    print(
        "Smoke configuration: "
        f"methods={','.join(selected)} agents={args.agents} tasks={args.tasks} "
        f"grid={args.width}x{args.height} steps={args.steps} episode_limit={args.episode_limit}",
        flush=True,
    )
    _print_problem_size(args)

    for method in selected:
        print(f"\n=== Smoke test: {method} ===", flush=True)
        run_method(method, args)

    print("\nAll requested smoke tests passed.", flush=True)


if __name__ == "__main__":
    main()
