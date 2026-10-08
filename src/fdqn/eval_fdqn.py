import sys
import os
import re
import csv
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stat_env.stat_gym import STATGymEnv  # Adjust this import as needed


# -------------------------
# FDQN model (same as training)
# -------------------------
HIDDEN_DIM = 32  # all runs use the same hidden dim


class FactorizedQNetwork(nn.Module):
    def __init__(self, input_dim, num_agents, num_sub_actions, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.agent_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_sub_actions) for _ in range(num_agents)]
        )

    def forward(self, state):
        # state: [B, D]
        x = self.shared_fc(state)  # [B, H]
        q_values = [head(x) for head in self.agent_heads]  # list of [B, A]
        return torch.stack(q_values, dim=1)  # [B, N, A]


class FDQNAgent:
    def __init__(self, obs_dim, num_agents, num_sub_actions, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[FDQNAgent] Using device: {self.device}", flush=True)

        self.num_agents = num_agents
        self.num_sub_actions = num_sub_actions

        self.q_net = FactorizedQNetwork(obs_dim, num_agents, num_sub_actions).to(self.device)
        self.q_net.eval()

    @torch.no_grad()
    def select_action(self, state_dict):
        """
        Greedy per-agent action with per-agent masks.
        Returns: list[int] length N
        """
        obs = torch.tensor(state_dict["obs"], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]
        masks = torch.tensor(state_dict["agent_masks"], dtype=torch.float32, device=self.device)     # [N, A]

        q = self.q_net(obs).squeeze(0)  # [N, A]
        q_masked = q.clone()
        q_masked[masks == 0] = -1e8
        actions = torch.argmax(q_masked, dim=1).tolist()
        return [int(a) for a in actions]


# -------------------------
# Helpers: locating models
# -------------------------
def extract_env_from_config(config: str):
    """
    config expected like: 3R_5V_5x5
    """
    m = re.match(r"^(\d+)R_(\d+)V_(\d+)x(\d+)$", config)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def extract_train_seed_from_path(path: str) -> str:
    m = re.search(r"(seed\d+)", path)
    return m.group(1) if m else "seedUNK"


def find_run_dir(fdqn_root: str, config: str, train_seed: int):
    exp_root = os.path.join(fdqn_root, "training_experiments", f"FDQN_{config}")
    if not os.path.isdir(exp_root):
        raise FileNotFoundError(f"Experiment folder not found: {exp_root}")

    seed_dirs = [d for d in os.listdir(exp_root) if d.startswith(f"seed{train_seed}_")]
    if len(seed_dirs) == 0:
        raise FileNotFoundError(f"No runs found for seed{train_seed} in {exp_root}")
    if len(seed_dirs) > 1:
        raise RuntimeError(
            f"Multiple runs found for seed{train_seed} in {exp_root}. "
            f"Please disambiguate:\n{seed_dirs}"
        )

    return os.path.join(exp_root, seed_dirs[0])


def load_weights_flex(model_dir: str, config: str, device):
    """
    Prefer weights-only (best for time-limited runs), then final model, then payload.
    Returns: (state_dict, selected_path)
    """
    candidates = [
        os.path.join(model_dir, "latest_weights.pth"),
        os.path.join(model_dir, f"{config}.pth"),
        os.path.join(model_dir, "latest_payload.pth"),
    ]

    for p in candidates:
        if not os.path.exists(p):
            continue

        obj = torch.load(p, map_location=device)

        # weights-only state_dict (dict of tensors)
        if isinstance(obj, dict) and "q_net" not in obj:
            return obj, p

        # payload checkpoint
        if isinstance(obj, dict) and "q_net" in obj:
            return obj["q_net"], p

    raise FileNotFoundError(f"No usable checkpoint found in {model_dir}. Tried: {candidates}")


# -------------------------
# Evaluation
# -------------------------
def run_eval(
    config: str,
    train_seed: int,
    num_bins: int,
    eval_seeds,
    episodes_per_seed: int = 1,
    max_steps_per_episode: int = 1600,
    device: str = "cuda",
):
    FDQN_ROOT = os.path.dirname(os.path.abspath(__file__))

    env_params = extract_env_from_config(config)
    if env_params is None:
        raise ValueError(f"Bad config format: {config} (expected like 3R_5V_5x5)")
    agents, tasks, width, height = env_params

    run_dir = find_run_dir(fdqn_root=FDQN_ROOT, config=config, train_seed=train_seed)
    model_dir = os.path.join(run_dir, "saved_models")

    # infer obs_dim
    env0 = STATGymEnv(seed=0, agents=agents, tasks=tasks, width=width, height=height,
                     policy=6, num_bins=num_bins)
    sample_state, _ = env0.reset()
    obs_dim = sample_state["obs"].shape[0]

    agent = FDQNAgent(obs_dim=obs_dim, num_agents=agents, num_sub_actions=(3 + tasks), device=device)

    weights, selected_path = load_weights_flex(model_dir=model_dir, config=config, device=agent.device)
    agent.q_net.load_state_dict(weights)
    agent.q_net.eval()

    train_seed_tag = extract_train_seed_from_path(run_dir)

    print(f"\n[Eval-FDQN] config={config} train_seed={train_seed_tag}")
    print(f"[Eval-FDQN] run_dir={run_dir}")
    print(f"[Eval-FDQN] loaded={selected_path}")
    print(f"[Eval-FDQN] eval_seeds={list(eval_seeds)} episodes_per_seed={episodes_per_seed}")

    eval_root = os.path.join(FDQN_ROOT, "evaluation", f"FDQN_{config}")
    os.makedirs(eval_root, exist_ok=True)
    out_csv = os.path.join(eval_root, f"eval_results_{train_seed_tag}.csv")
    out_summary = os.path.join(eval_root, f"eval_summary_{train_seed_tag}.txt")

    rows = []
    for eval_seed in eval_seeds:
        for ep_idx in range(episodes_per_seed):
            sd = int(eval_seed) * 100000 + int(ep_idx)
            np.random.seed(sd)
            random.seed(sd)
            torch.manual_seed(sd)

            env = STATGymEnv(seed=eval_seed, agents=agents, tasks=tasks, width=width, height=height,
                            policy=6, num_bins=num_bins)
            state, _ = env.reset()

            done = False
            steps = 0
            ep_reward = 0.0
            forced_idle_sum = 0
            num_conflicts_sum = 0
            unique_assigned_list = []

            while not done and steps < max_steps_per_episode:
                action = agent.select_action(state_dict=state)
                next_state, reward, done, _, info = env.step(action)
                state = next_state

                ep_reward += float(reward)
                steps += 1

                if "forced_idle" in info:
                    forced_idle_sum += int(info["forced_idle"])
                if "num_conflicts" in info:
                    num_conflicts_sum += int(info["num_conflicts"])
                if "unique_tasks_assigned" in info:
                    unique_assigned_list.append(int(info["unique_tasks_assigned"]))

            unique_assigned_mean = float(np.mean(unique_assigned_list)) if unique_assigned_list else 0.0

            rows.append({
                "config": config,
                "train_seed": train_seed_tag,
                "eval_seed": int(eval_seed),
                "episode_idx": int(ep_idx),
                "episode_reward": float(ep_reward),
                "steps": int(steps),
                "done": int(done),
                "forced_idle_sum": int(forced_idle_sum),
                "num_conflicts_sum": int(num_conflicts_sum),
                "unique_tasks_assigned_mean": float(unique_assigned_mean),
            })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rewards = np.array([r["episode_reward"] for r in rows], dtype=float)
    steps_arr = np.array([r["steps"] for r in rows], dtype=float)

    with open(out_summary, "w") as f:
        f.write("FDQN Evaluation Summary\n")
        f.write(f"config: {config}\n")
        f.write(f"train_seed: {train_seed_tag}\n")
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"loaded_checkpoint: {selected_path}\n")
        f.write(f"num_bins: {num_bins}\n")
        f.write(f"eval_seeds: {list(eval_seeds)}\n")
        f.write(f"episodes_per_seed: {episodes_per_seed}\n\n")
        f.write(f"Total episodes: {len(rows)}\n")
        f.write(f"Reward mean={rewards.mean():.3f} std={rewards.std():.3f} "
                f"min={rewards.min():.3f} max={rewards.max():.3f}\n")
        f.write(f"Steps  mean={steps_arr.mean():.3f} std={steps_arr.std():.3f} "
                f"min={steps_arr.min():.3f} max={steps_arr.max():.3f}\n")

    print(f"[Eval-FDQN] Saved: {out_csv}")
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="e.g., 3R_5V_5x5")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--eval_seeds", type=int, nargs="+", default=list(range(100, 120)))
    parser.add_argument("--episodes_per_seed", type=int, default=1)
    parser.add_argument("--max_steps_per_episode", type=int, default=1600)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_eval(
        config=args.config,
        train_seed=args.train_seed,
        num_bins=args.num_bins,
        eval_seeds=args.eval_seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_steps_per_episode=args.max_steps_per_episode,
        device=args.device,
    )

if __name__ == "__main__":
    main()
