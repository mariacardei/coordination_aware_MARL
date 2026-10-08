import sys
import os

import re
import csv
import argparse
import random
from itertools import product

import numpy as np
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stat_env.stat_model import DEBUG
from stat_env.stat_gym import STATGymEnv  # Adjust this import as needed

# -------------------------
# DQN model (same as training)
# -------------------------
class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, joint_action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, joint_action_dim),
        )

    def forward(self, state):
        return self.net(state)


class DQNAgent:
    def __init__(self, obs_dim, joint_actions_np, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Using device: {self.device}", flush=True)

        self.JOINT_ACTIONS_NP = np.array(joint_actions_np, dtype=np.int64)  # [J, N]
        self.J = self.JOINT_ACTIONS_NP.shape[0]

        self.q_net = DeepQNetwork(obs_dim, self.J).to(self.device)
        self.q_net.eval()

    def _joint_valid_mask(self, agent_masks_np):
        N, A = agent_masks_np.shape
        valid_matrix = agent_masks_np[np.arange(N)[:, None], self.JOINT_ACTIONS_NP.T]  # [N, J]
        joint_valid = valid_matrix.all(axis=0).astype(np.float32)  # [J]
        return torch.from_numpy(joint_valid).to(self.device)       # [J]

    @torch.no_grad()
    def select_action(self, state_dict, eval_mode=True):
        obs = torch.tensor(state_dict["obs"], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]
        agent_masks_np = np.array(state_dict["agent_masks"])

        q_all = self.q_net(obs).squeeze(0)  # [J]
        joint_valid = self._joint_valid_mask(agent_masks_np)  # [J]
        masked_q = q_all.clone()
        masked_q[joint_valid == 0] = -1e8

        idx = int(torch.argmax(masked_q).item())
        return self.JOINT_ACTIONS_NP[idx].tolist()


def extract_env_from_filename(filename: str):
    m = re.match(r"(\d+)R_(\d+)V_(\d+)x(\d+)\.pth$", filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def extract_train_seed_from_path(model_path: str) -> str:
    # grabs seed0 / seed1 / seed2 from the run directory if present
    m = re.search(r"(seed\d+)", model_path)
    return m.group(1) if m else "seedUNK"

def find_trained_model(dqn_root, config, train_seed):
    """
    Finds:
    DQN/training_experiments/DQN_<config>/seed<train_seed>_*/saved_models/<config>.pth
    """
    exp_root = os.path.join(
        dqn_root,
        "training_experiments",
        f"DQN_{config}"
    )

    if not os.path.isdir(exp_root):
        raise FileNotFoundError(f"Experiment folder not found: {exp_root}")

    seed_dirs = [
        d for d in os.listdir(exp_root)
        if d.startswith(f"seed{train_seed}_")
    ]

    if len(seed_dirs) == 0:
        raise FileNotFoundError(f"No runs found for seed{train_seed} in {exp_root}")
    if len(seed_dirs) > 1:
        raise RuntimeError(
            f"Multiple runs found for seed{train_seed}. "
            f"Please disambiguate:\n{seed_dirs}"
        )

    run_dir = os.path.join(exp_root, seed_dirs[0])
    model_path = os.path.join(
        run_dir,
        "saved_models",
        f"{config}.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return model_path

def run_eval(
    config: str,
    train_seed: int,
    num_bins: int,
    eval_seeds,
    episodes_per_seed: int = 1,
    max_steps_per_episode: int = 1600,
    device: str = "cuda",
):
    # DQN root = directory containing this file
    DQN_ROOT = os.path.dirname(os.path.abspath(__file__))

    model_path = find_trained_model(
        dqn_root=DQN_ROOT,
        config=config,
        train_seed=train_seed
    )

    env_params = extract_env_from_filename(f"{config}.pth")
    if env_params is None:
        raise ValueError(f"Bad config format: {config} (expected like 3R_5V_5x5)")

    agents, tasks, width, height = env_params
    train_seed_tag = extract_train_seed_from_path(model_path)

    print(f"\n[Eval] config={config} train_seed={train_seed_tag}")
    print(f"[Eval] model={model_path}")
    print(f"[Eval] eval_seeds={list(eval_seeds)} episodes_per_seed={episodes_per_seed}")

    # Build joint action list
    num_sub_actions = 3 + tasks
    JOINT_ACTIONS = list(product(range(num_sub_actions), repeat=agents))

    # Get obs_dim
    env0 = STATGymEnv(seed=0, agents=agents, tasks=tasks, width=width, height=height,
                     policy=6, num_bins=num_bins)
    sample_state, _ = env0.reset()
    obs_dim = sample_state["obs"].shape[0]

    # Load agent
    agent = DQNAgent(obs_dim=obs_dim, joint_actions_np=JOINT_ACTIONS, device=device)
    weights = torch.load(model_path, map_location=agent.device)
    agent.q_net.load_state_dict(weights)
    agent.q_net.eval()

    # Output dir
    eval_root = os.path.join(DQN_ROOT, "evaluation", f"DQN_{config}")
    os.makedirs(eval_root, exist_ok=True)
    out_csv = os.path.join(eval_root, f"eval_results_{train_seed_tag}.csv")
    out_summary = os.path.join(eval_root, f"eval_summary_{train_seed_tag}.txt")

    rows = []
    for eval_seed in eval_seeds:
        for ep_idx in range(episodes_per_seed):
            sd = eval_seed * 100000 + ep_idx
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
                action = agent.select_action(state_dict=state, eval_mode=True)
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

    # Save CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary
    rewards = np.array([r["episode_reward"] for r in rows], dtype=float)
    steps_arr = np.array([r["steps"] for r in rows], dtype=float)
    with open(out_summary, "w") as f:
        f.write("DQN Evaluation Summary\n")
        f.write(f"config: {config}\n")
        f.write(f"train_seed: {train_seed_tag}\n")
        f.write(f"model: {model_path}\n")
        f.write(f"num_bins: {num_bins}\n")
        f.write(f"eval_seeds: {list(eval_seeds)}\n")
        f.write(f"episodes_per_seed: {episodes_per_seed}\n\n")
        f.write(f"Total episodes: {len(rows)}\n")
        f.write(f"Reward mean={rewards.mean():.3f} std={rewards.std():.3f} "
                f"min={rewards.min():.3f} max={rewards.max():.3f}\n")
        f.write(f"Steps  mean={steps_arr.mean():.3f} std={steps_arr.std():.3f} "
                f"min={steps_arr.min():.3f} max={steps_arr.max():.3f}\n")

    print(f"[Eval] Saved: {out_csv}")
    return out_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
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
