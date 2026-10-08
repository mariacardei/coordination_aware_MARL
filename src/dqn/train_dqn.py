import sys
import os
import time
import csv
import argparse
import random
from collections import deque
from itertools import product
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stat_env.stat_model import DEBUG
from stat_env.stat_gym import STATGymEnv  # Adjust this import as needed

torch.backends.cudnn.benchmark = True

'''
Vanilla (centralized) DQN over JOINT actions
- Joint action space J = (3 + tasks) ^ agents
- Network outputs Q(s, a_joint) for all joint actions
- Masking: builds a joint-valid mask from per-agent masks each step
To run ex.: python train_dqn.py --agents 3 --tasks 5 --width 25 --height 15 --num_bins 5 --episodes 1000
'''


class Tee:
    """Write to both console and a file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

# Deep Q Network (no factorization)
class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, joint_action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, joint_action_dim)
        )
    def forward(self, state):  # state: [B, input_dim]
        return self.net(state) # [B, J]



# Vanilla DQN Agent

# DQN Agent over joint actions
class DQNAgent:
    def __init__(self, obs_dim, joint_actions_np,
                 lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=300000,
                 buffer_capacity=100000, batch_size=128,
                 target_update_freq=5000, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Using device: {self.device}", flush=True)

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Joint action lookup (both numpy + torch)
        self.JOINT_ACTIONS_NP = np.array(joint_actions_np, dtype=np.int64)  # [J, N]
        self.J = self.JOINT_ACTIONS_NP.shape[0]
        self.JOINT_ACTIONS_T = torch.as_tensor(self.JOINT_ACTIONS_NP, dtype=torch.long, device=self.device)  # [J, N]

        # Q-networks
        self.q_net = DeepQNetwork(obs_dim, self.J).to(self.device)
        self.target_net = DeepQNetwork(obs_dim, self.J).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay = deque(maxlen=buffer_capacity)

    # def _epsilon(self):
    #     # same slow log-style schedule used
    #     return self.epsilon_final + (self.epsilon_start - self.epsilon_final) / (1 + np.log1p(self.steps_done / self.epsilon_decay))

    def _epsilon(self):
        """
        Linear epsilon decay from epsilon_start -> epsilon_final over epsilon_decay env steps.
        After epsilon_decay steps, epsilon stays at epsilon_final.
        """
        frac = min(1.0, self.steps_done / float(self.epsilon_decay))
        return self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)

    def _joint_valid_mask(self, agent_masks_np):
        """
        Build joint-valid mask from per-agent masks.
        agent_masks_np: [N, A] (0/1)
        returns torch.FloatTensor [J] (1.0 for valid, 0.0 for invalid)
        A joint action is valid iff every agent's selected action is valid.
        """
        N, A = agent_masks_np.shape
        # For each joint index j, check mask[i, action_i_j] == 1 for all agents i
        # JOINT_ACTIONS_NP.T: [N, J]
        valid_matrix = agent_masks_np[np.arange(N)[:, None], self.JOINT_ACTIONS_NP.T]  # [N, J]
        joint_valid = valid_matrix.all(axis=0).astype(np.float32)  # [J]
        return torch.from_numpy(joint_valid).to(self.device)       # [J]

    def select_action(self, state_dict, eval_mode=False):
        """
        Return:
          joint_tuple: list[int] of length N (env-consumable)
          joint_idx:   int index into JOINT_ACTIONS_NP
        """
        obs = torch.tensor(state_dict["obs"], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]
        agent_masks_np = state_dict["agent_masks"]  # [N, A], numpy or list->np

        with torch.no_grad():
            q_all = self.q_net(obs).squeeze(0)  # [J]
        joint_valid = self._joint_valid_mask(np.array(agent_masks_np))  # [J]
        masked_q = q_all.clone()
        masked_q[joint_valid == 0] = -1e8

        if eval_mode:
            idx = int(torch.argmax(masked_q).item())
        else:
            eps = self._epsilon()
            self.steps_done += 1
            if random.random() < eps:
                valid_idxs = (joint_valid == 1).nonzero(as_tuple=False).squeeze(1)
                if len(valid_idxs) == 0:
                    idx = 0
                else:
                    ridx = torch.randint(len(valid_idxs), (1,), device=self.device).item()
                    idx = int(valid_idxs[ridx].item())
            else:
                idx = int(torch.argmax(masked_q).item())

        joint_tuple = self.JOINT_ACTIONS_NP[idx].tolist()  # list of length N
        return joint_tuple, idx

    def store_transition(self, state, joint_idx, reward, next_state, done):
        self.replay.append((state, joint_idx, reward, next_state, done))

    def _sample_batch(self):
        batch = random.sample(self.replay, self.batch_size)
        s, aidx, r, ns, d = map(list, zip(*batch))
        return s, aidx, r, ns, d

    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        s, aidx, r, ns, d = self._sample_batch()

        s_obs  = torch.tensor(np.array([x["obs"] for x in s]),  dtype=torch.float32, device=self.device)  # [B, D]
        ns_obs = torch.tensor(np.array([x["obs"] for x in ns]), dtype=torch.float32, device=self.device)  # [B, D]

        s_masks  = np.array([x["agent_masks"] for x in s])   # [B, N, A] numpy
        ns_masks = np.array([x["agent_masks"] for x in ns])  # [B, N, A] numpy

        aidx = torch.tensor(aidx, dtype=torch.long, device=self.device).unsqueeze(1)      # [B,1]
        r    = torch.tensor(r,    dtype=torch.float32, device=self.device).unsqueeze(1)   # [B,1]
        d    = torch.tensor(d,    dtype=torch.float32, device=self.device).unsqueeze(1)   # [B,1]

        # Current Q(s, a_joint)
        q_all = self.q_net(s_obs)                 # [B, J]
        q_sa  = q_all.gather(1, aidx)             # [B, 1]

        # Target: max over VALID joint actions in next state
        with torch.no_grad():
            q_next_all = self.target_net(ns_obs)  # [B, J]

            # Build joint masks per sample and apply
            joint_masks_list = []
            for bm in ns_masks:  # bm: [N, A]
                jm = self._joint_valid_mask(bm).unsqueeze(0)  # [1, J]
                joint_masks_list.append(jm)
            joint_masks = torch.cat(joint_masks_list, dim=0)  # [B, J]

            q_next_all = q_next_all.masked_fill(joint_masks == 0, -1e8)
            q_next_max = q_next_all.max(dim=1, keepdim=True).values  # [B,1]
            target = r + self.gamma * q_next_max * (1 - d)

        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

def make_dqn_agent(env, args):
    sample_state, _ = env.reset()
    obs_dim = sample_state["obs"].shape[0]
    num_agents = args.agents
    num_sub = 3 + args.tasks
    JOINT_ACTIONS = list(product(range(num_sub), repeat=num_agents))
    
    return DQNAgent(
        obs_dim=obs_dim,
        joint_actions_np=JOINT_ACTIONS,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_final=args.epsilon_final,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq
    )

def _safe_env_reset(env, seed=None):
    """
    Tries to reset with a seed (newer Gym API), otherwise falls back.
    Returns (state, info_or_none).
    """
    if seed is None:
        return env.reset()

    try:
        return env.reset(seed=seed)
    except TypeError:
        pass

    if hasattr(env, "set_seed"):
        try:
            env.set_seed(seed)
        except Exception:
            pass

    return env.reset()


def _as_int_actions(joint_action):
    """
    Ensure env receives pure Python ints (fixes 'num_conflicts always 0' bug
    when env logic does isinstance(x, int) or uses dict keys / comparisons).
    Supports list/np arrays/torch tensors/np scalar types.
    """
    if isinstance(joint_action, torch.Tensor):
        joint_action = joint_action.detach().cpu().tolist()
    return [int(x) for x in joint_action]


def _run_one_eval_episode_dqn(env, agent, state, max_steps):
    """
    Greedy episode for DQN. Uses agent.select_action(..., eval_mode=True).
    Returns: (episode_reward, steps_in_episode, coord_metrics dict)
    """
    episode_reward = 0.0
    steps_in_episode = 0

    J_upper_list = []
    forced_idle_sum = 0
    unique_assigned_list = []
    deterministic_agents_list = []
    num_conflicts_sum = 0

    ep_start = time.time()
    for _t in range(max_steps):
        joint_action, _joint_idx = agent.select_action(state, eval_mode=True)
        joint_action = _as_int_actions(joint_action)

        next_state, reward, done, _, info = env.step(joint_action)

        if "J_upper" in info:
            J_upper_list.append(info["J_upper"])
        if "forced_idle" in info:
            forced_idle_sum += int(info["forced_idle"])
        if "unique_tasks_assigned" in info:
            unique_assigned_list.append(int(info["unique_tasks_assigned"]))
        if "deterministic_agents" in info:
            deterministic_agents_list.append(int(info["deterministic_agents"]))
        if "num_conflicts" in info:
            num_conflicts_sum += int(info["num_conflicts"])

        episode_reward += float(reward)
        steps_in_episode += 1
        state = next_state

        if done:
            break

    ep_time = time.time() - ep_start
    steps_per_sec = steps_in_episode / max(ep_time, 1e-6)

    coord = {
        "J_upper_median": float(np.median(J_upper_list)) if J_upper_list else 0.0,
        "ForcedIdle_sum": int(forced_idle_sum),
        "UniqueTasksAssigned_mean": float(np.mean(unique_assigned_list)) if unique_assigned_list else 0.0,
        "DeterministicAgents_mean": float(np.mean(deterministic_agents_list)) if deterministic_agents_list else 0.0,
        "NumConflicts_sum": int(num_conflicts_sum),
        "Time_sec": float(ep_time),
        "StepsPerSec": float(steps_per_sec),
    }
    return float(episode_reward), int(steps_in_episode), coord


def train_dqn(
    env,
    agent,
    experiment_dir,
    args,
    total_steps_limit=1000000,
    max_steps_per_episode=1000,
    update_every=4,
    save_every_steps=5000,
    eval_every_steps=10000,
    eval_episodes=20,
    eval_max_steps=1000,
    train_seed=0,
    exp_name=None,
    eval_during_training=False,
):
    total_steps = 0
    episode = 0
    run_start_time = time.time()

    if exp_name is None:
        exp_name = "DQN"

    os.makedirs(experiment_dir, exist_ok=True)
    saved_models_dir = os.path.join(experiment_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    # TensorBoard
    tb_log_dir = os.path.join(experiment_dir, "tb_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)

    # CSV
    csv_dir = os.path.join(experiment_dir, "csv_output")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "train_results.csv")
    last_eval_T = -eval_every_steps - 1  # force first eval immediately (PyMARL style)

    # ----- TRAIN CSV -----
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "TotalSteps","Episode",
            "Reward","StepsInEpisode","AvgLoss","Epsilon","Time_sec","WallTimeTotal_sec",
            "J_upper_median","ForcedIdle_sum","UniqueTasksAssigned_mean",
            "DeterministicAgents_mean","NumConflicts_sum","StepsPerSec"
        ])

    # ----- EVAL CSVs (only if enabled) -----
    eval_csv_path = os.path.join(csv_dir, "eval_results.csv")
    eval_sum_path = os.path.join(csv_dir, "eval_summary.csv")

    if eval_during_training:
        with open(eval_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "TotalSteps","TrainEpisode","EvalEpisodeIdx","EvalSeed",
                "Reward","StepsInEpisode","Epsilon","Time_sec",
                "J_upper_median","ForcedIdle_sum","UniqueTasksAssigned_mean",
                "DeterministicAgents_mean","NumConflicts_sum","StepsPerSec"
            ])

        with open(eval_sum_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "TotalSteps","TrainEpisode","NEvalEpisodes",
                "Reward_mean","Reward_std",
                "Steps_mean","Steps_std",
                "J_upper_median_mean","J_upper_median_std",
                "ForcedIdle_sum_mean","ForcedIdle_sum_std",
                "UniqueTasksAssigned_mean","UniqueTasksAssigned_std",
                "DeterministicAgents_mean","DeterministicAgents_std",
                "NumConflicts_sum_mean","NumConflicts_sum_std",
            ])

    # Separate eval env (prevents contamination / makes resets deterministic)
    eval_env = None
    if eval_during_training:
        eval_env = STATGymEnv(
            seed=int(train_seed) + 999999,
            agents=args.agents,
            tasks=args.tasks,
            width=args.width,
            height=args.height,
            policy=6,
            num_bins=args.num_bins,
        )

    while total_steps < total_steps_limit:
        episode_reward = 0.0
        steps_in_episode = 0
        losses = []
        ep_start_time = time.time()

        J_upper_list = []
        forced_idle_sum = 0
        unique_assigned_list = []
        deterministic_agents_list = []
        num_conflicts_sum = 0

        try:
            state, _ = env.reset()
        except Exception as e:
            print(f"[ERROR] env.reset() failed at total_steps={total_steps}: {e}", flush=True)
            continue

        for _t in range(max_steps_per_episode):
            joint_action, joint_idx = agent.select_action(state)
            joint_action = _as_int_actions(joint_action)  # critical for NumConflicts correctness
            next_state, reward, done, _, info = env.step(joint_action)

            if "J_upper" in info:
                J_upper_list.append(info["J_upper"])
            if "forced_idle" in info:
                forced_idle_sum += int(info["forced_idle"])
            if "unique_tasks_assigned" in info:
                unique_assigned_list.append(int(info["unique_tasks_assigned"]))
            if "deterministic_agents" in info:
                deterministic_agents_list.append(int(info["deterministic_agents"]))
            if "num_conflicts" in info:
                num_conflicts_sum += int(info["num_conflicts"])

            agent.store_transition(state, joint_idx, reward, next_state, done)

            state = next_state
            episode_reward += float(reward)
            total_steps += 1
            steps_in_episode += 1

            if total_steps > 0 and (total_steps % save_every_steps == 0):
                torch.save(agent.q_net.state_dict(), os.path.join(saved_models_dir, "latest_weights.pth"))

            if total_steps % update_every == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

            if total_steps % agent.target_update_freq == 0:
                agent.update_target_network()

            if done or total_steps >= total_steps_limit:
                break

        # === eval during training (between episodes) ===
        if eval_during_training and (total_steps - last_eval_T >= eval_every_steps) and (total_steps < total_steps_limit):
            last_eval_T = total_steps

            agent.q_net.eval()
            try:
                with torch.no_grad():
                    eval_rewards, eval_steps, eval_coord = [], [], []

                    for i in range(eval_episodes):
                        eval_seed = int(train_seed) * 1000 + int(i) 
                        eval_state, _ = _safe_env_reset(eval_env, seed=eval_seed)

                        r, s, coord = _run_one_eval_episode_dqn(
                            eval_env, agent, eval_state, max_steps=eval_max_steps
                        )

                        eval_rewards.append(r)
                        eval_steps.append(s)
                        eval_coord.append(coord)

                        with open(eval_csv_path, "a", newline="") as f:
                            w = csv.writer(f)
                            w.writerow([
                                int(total_steps),
                                int(episode),
                                int(i),
                                int(eval_seed),
                                float(r),
                                int(s),
                                0.0,
                                float(coord["Time_sec"]),
                                float(coord["J_upper_median"]),
                                int(coord["ForcedIdle_sum"]),
                                float(coord["UniqueTasksAssigned_mean"]),
                                float(coord["DeterministicAgents_mean"]),
                                int(coord["NumConflicts_sum"]),
                                float(coord["StepsPerSec"]),
                            ])

                    rets = np.asarray(eval_rewards, dtype=float)
                    stps = np.asarray(eval_steps, dtype=float)

                    def _mean(key): return float(np.mean([d[key] for d in eval_coord]))
                    def _std(key):
                        vals = [d[key] for d in eval_coord]
                        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

                    with open(eval_sum_path, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([
                            int(total_steps),
                            int(episode),
                            int(eval_episodes),
                            float(np.mean(rets)), float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0,
                            float(np.mean(stps)), float(np.std(stps, ddof=1)) if len(stps) > 1 else 0.0,
                            _mean("J_upper_median"), _std("J_upper_median"),
                            _mean("ForcedIdle_sum"), _std("ForcedIdle_sum"),
                            _mean("UniqueTasksAssigned_mean"), _std("UniqueTasksAssigned_mean"),
                            _mean("DeterministicAgents_mean"), _std("DeterministicAgents_mean"),
                            _mean("NumConflicts_sum"), _std("NumConflicts_sum"),
                        ])

                    tb_writer.add_scalar("eval/Reward_mean", float(np.mean(rets)), total_steps)
                    tb_writer.add_scalar("eval/Reward_std",  float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0, total_steps)
                    tb_writer.add_scalar("eval/Steps_mean",  float(np.mean(stps)), total_steps)
                    tb_writer.add_scalar("eval/Steps_std",   float(np.std(stps, ddof=1)) if len(stps) > 1 else 0.0, total_steps)

                    tb_writer.add_scalar("eval/J_upper_median_mean", _mean("J_upper_median"), total_steps)
                    tb_writer.add_scalar("eval/ForcedIdle_sum_mean", _mean("ForcedIdle_sum"), total_steps)
                    tb_writer.add_scalar("eval/UniqueTasksAssigned_mean", _mean("UniqueTasksAssigned_mean"), total_steps)
                    tb_writer.add_scalar("eval/DeterministicAgents_mean", _mean("DeterministicAgents_mean"), total_steps)
                    tb_writer.add_scalar("eval/NumConflicts_sum_mean", _mean("NumConflicts_sum"), total_steps)

                    print(
                        f"[EVAL] total_steps={total_steps} | "
                        f"R_mean={np.mean(rets):.3f} R_std={(np.std(rets, ddof=1) if len(rets)>1 else 0.0):.3f} | "
                        f"Steps_mean={np.mean(stps):.1f}",
                        flush=True,
                    )
            finally:
                agent.q_net.train()

        # --- train logging ---
        epsilon_val = agent._epsilon()
        ep_secs = round(time.time() - ep_start_time, 2)

        J_upper_median = float(np.median(J_upper_list)) if J_upper_list else 0.0
        unique_assigned_mean = float(np.mean(unique_assigned_list)) if unique_assigned_list else 0.0
        deterministic_agents_mean = float(np.mean(deterministic_agents_list)) if deterministic_agents_list else 0.0
        steps_per_sec = (steps_in_episode / ep_secs) if ep_secs > 0 else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        tb_writer.add_scalar('combinatorics/J_upper_median', J_upper_median, total_steps)
        tb_writer.add_scalar('coord/forced_idle_sum', forced_idle_sum, total_steps)
        tb_writer.add_scalar('coord/unique_tasks_assigned_mean', unique_assigned_mean, total_steps)
        tb_writer.add_scalar('coord/deterministic_agents_mean', deterministic_agents_mean, total_steps)
        tb_writer.add_scalar('coord/num_conflicts_sum', num_conflicts_sum, total_steps)
        tb_writer.add_scalar('sys/steps_per_sec', steps_per_sec, total_steps)
        tb_writer.add_scalar("sys/episode", episode, total_steps)

        tb_writer.add_scalar('Episode Reward', episode_reward, total_steps)
        tb_writer.add_scalar('Total Steps Per Episode', steps_in_episode, total_steps)
        tb_writer.add_scalar('Average Loss', avg_loss, total_steps)
        tb_writer.add_scalar('Epsilon', epsilon_val, total_steps)

        wall_time_total_sec = float(time.time() - run_start_time)

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                total_steps, episode,
                float(episode_reward), int(steps_in_episode), float(avg_loss), float(epsilon_val), float(ep_secs), float(wall_time_total_sec),
                float(J_upper_median), int(forced_idle_sum), float(unique_assigned_mean),
                float(deterministic_agents_mean), int(num_conflicts_sum), float(steps_per_sec)
            ])

        print(
            f"Exp: {exp_name}, Episode {episode + 1} Done -- "
            f"Reward: {episode_reward}, Steps: {steps_in_episode}, Avg loss = {avg_loss}, "
            f"TotalSteps={total_steps}/{total_steps_limit}",
            flush=True,
        )

        episode += 1

        if (episode) % 1000 == 0:
            checkpoint_path = os.path.join(saved_models_dir, f"checkpoint_{episode}.pth")
            torch.save(agent.q_net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at episode {episode}", flush=True)

    tb_writer.close()
    try:
        if eval_env is not None:
            eval_env.close()
    except Exception:
        pass

    return agent


# Main Function
if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", dest="agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--tasks", dest="tasks", type=int, default=3, help="Number of tasks")
    parser.add_argument("--width", type=int, default=5, help="Grid width")
    parser.add_argument("--height", type=int, default=5, help="Grid height")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of distance bins") #keep 5 for experiments here
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # hyperparameters:
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for DQN updates")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target_update_freq", type=int, default=5000, 
                        help="How many steps between target network updates")
    # parser.add_argument("--epsilon_decay", type=float, default=300000, 
    #                     help="Number of env steps over which epsilon decays from 1.0 to epsilon_final")
    parser.add_argument("--epsilon_final", type=float, default=0.05,
                        help="Final epsilon value after decay")
    parser.add_argument("--epsilon_frac", type=float, default=0.3,
                        help="Fraction of total training steps over which epsilon decays (0-1)")
    parser.add_argument("--buffer_capacity", type=int, default=100000,
                        help="Replay buffer capacity")
    parser.add_argument("--total_steps", type=int, default=1000000,
                        help="Total env steps training budget (upper bound)")
    parser.add_argument("--epsilon_decay_steps", type=int, default=None,
                        help="Override: number of env steps to linearly decay epsilon from 1.0 to epsilon_final. "
                        "If not set, uses epsilon_frac * total_steps.")

    # --- to eval during training ---
    parser.add_argument("--save_every_steps", type=int, default=10000,
                        help="Save latest weights every N env steps (QMIX save_model_interval).")

    parser.add_argument("--eval_during_training", action="store_true",
                        help="If set, run greedy evaluation periodically during training.")

    parser.add_argument("--eval_every_steps", type=int, default=10000,
                        help="Run eval every N env steps (QMIX test_interval).")

    parser.add_argument("--eval_episodes", type=int, default=20,
                        help="Number of greedy eval episodes per eval point (QMIX test_nepisode).")

    parser.add_argument("--eval_max_steps", type=int, default=1000,
                        help="Max steps per eval episode (match env horizon).")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory for training outputs. Defaults to results/training at the repository root.",
    )


    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # convert epsilon fraction into absolute step horizon
    # epsilon decay horizon (env steps)
    if args.epsilon_decay_steps is not None:
        args.epsilon_decay = max(1, int(args.epsilon_decay_steps))
    else:
        args.epsilon_decay = max(1, int(args.epsilon_frac * args.total_steps))

    decay_src = f"override({args.epsilon_decay_steps})" if args.epsilon_decay_steps is not None else "epsilon_frac*total_steps"
    print(
        f"Running with agents={args.agents}, tasks={args.tasks}, width={args.width}, height={args.height}, bins={args.num_bins} | "
        f"batch_size={args.batch_size}, lr={args.lr}, gamma={args.gamma}, target_update_freq={args.target_update_freq}, "
        f"epsilon_final={args.epsilon_final}, epsilon_decay_steps={args.epsilon_decay} [{decay_src}], "
        f"epsilon_frac={args.epsilon_frac}, buffer_capacity={args.buffer_capacity}, total_steps={args.total_steps}, seed={args.seed}",
        flush=True,
    )



    # Create the environment.
    env = STATGymEnv(seed=args.seed, agents=args.agents, tasks=args.tasks,
                    width=args.width, height=args.height, policy=6, num_bins=args.num_bins)

    # Create the agent.
    agent = make_dqn_agent(env, args)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # this file's folder
    REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    output_root = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(REPO_ROOT, "results", "training")

    base_dir = os.path.join(
        output_root,
        f"DQN_{args.agents}agents_{args.tasks}tasks_{args.width}x{args.height}"
    )
    os.makedirs(base_dir, exist_ok=True)


    experiment_name = (
        f"seed{args.seed}_{args.num_bins}bins_{args.total_steps}steps_{args.lr}lr_{args.gamma}gamma_"
        f"{args.target_update_freq}targetupdate_{args.batch_size}batch_"
        f"{args.epsilon_frac}epsfrac_{int(args.epsilon_decay)}epsdecay_"
        f"{args.epsilon_final}epsfinal"
    )


    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # ---- Per-run logs folder ----
    logs_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, "stdout_stderr.log")
    log_f = open(log_path, "a", buffering=1)  # line-buffered

    # header
    log_f.write(f"\n===== Run started {datetime.now().isoformat()} =====\n")

    # tee prints to both terminal + file
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    print(f"[Logging] Writing stdout/stderr to: {log_path}", flush=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Train
    start_time = time.time()
    exp_name = f"DQN_{args.agents}agents_{args.tasks}tasks_{args.width}x{args.height}"
    
    trained_agent = train_dqn(
        env,
        agent,
        experiment_dir,
        args,
        total_steps_limit=args.total_steps,
        max_steps_per_episode=1000,   
        update_every=4,
        save_every_steps=args.save_every_steps,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        train_seed=args.seed,
        exp_name=exp_name,
        eval_during_training=args.eval_during_training,
    )

    secs = round(time.time() - start_time, 2)
    print(f"Training finished in {secs} seconds, {secs/60:.2f} minutes.", flush=True)

    # Save final model
    saved_models_dir = os.path.join(experiment_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    final_model_file = os.path.join(saved_models_dir, f"{args.agents}agents_{args.tasks}tasks_{args.width}x{args.height}.pth")
    torch.save(trained_agent.q_net.state_dict(), final_model_file)
    torch.save(trained_agent.q_net.state_dict(), os.path.join(saved_models_dir, "latest_weights.pth"))
    print(f"Final model saved at: {final_model_file}")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_f.close()
