"""PyMARL integration for STAT."""

from __future__ import annotations

import random

import numpy as np

from .config import STATConfig
from .core import STATModel, get_distance

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for environment-only use.
    torch = None


class STATPyMARLEnv:
    """PyMARL-compatible STAT environment.

    The API mirrors the subset PyMARL expects: reset/step, global state,
    per-agent observations, available action masks, and environment metadata.
    """

    def __init__(
        self,
        seed: int = 0,
        agents: int = 2,
        tasks: int = 3,
        width: int = 5,
        height: int = 5,
        num_bins: int = 5,
        episode_limit: int = 1600,
        agent_speeds=None,
    ):
        self.config = STATConfig(
            agents=agents,
            tasks=tasks,
            width=width,
            height=height,
            num_bins=num_bins,
            seed=seed,
            episode_limit=episode_limit,
            agent_speeds=tuple(agent_speeds) if agent_speeds is not None else None,
        )

        self.seed = int(seed)
        self.n_agents = self.config.agents
        self.n_tasks = self.config.tasks
        self.width = self.config.width
        self.height = self.config.height
        self.num_bins = self.config.num_bins
        self.episode_limit = self.config.episode_limit
        self.n_actions = self.config.per_agent_actions
        self.t = 0
        self.agent_speeds = list(self.config.speeds) if self.config.speeds else [1.0] * self.n_agents
        self.model = self._make_model(self.seed)
        self._reset_episode_stats()

    @classmethod
    def from_config(cls, config: STATConfig) -> "STATPyMARLEnv":
        return cls(
            seed=config.seed,
            agents=config.agents,
            tasks=config.tasks,
            width=config.width,
            height=config.height,
            num_bins=config.num_bins,
            episode_limit=config.episode_limit,
            agent_speeds=list(config.speeds) if config.speeds else None,
        )

    def _make_model(self, seed: int) -> STATModel:
        return STATModel(
            seed,
            self.n_agents,
            self.n_tasks,
            self.width,
            self.height,
            policy=6,
            num_bins=self.num_bins,
            agent_speeds=self.agent_speeds,
        )

    def _reset_episode_stats(self) -> None:
        self.ep_steps = 0
        self.ep_reward = 0.0
        self.ep_forced_idle = 0
        self.ep_num_conflicts = 0
        self.ep_J_upper = []
        self.ep_unique_assigned = []
        self.ep_deterministic_agents = []

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self.model = self._make_model(self.seed)

    def reset(self, seed=None):
        if seed is not None:
            self.set_seed(seed)
        else:
            self.model.reset()
        self.t = 0
        self._reset_episode_stats()
        return

    def step(self, actions):
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions).reshape(-1)
        actions = [int(x) for x in actions.tolist()]
        final_actions = actions[:]

        avail = self.get_avail_actions()
        valid_per_agent = avail.sum(axis=1).astype(np.int32)
        j_upper = int(np.prod(valid_per_agent)) if valid_per_agent.size > 0 else 0
        deterministic_agents = int((valid_per_agent == 1).sum())

        task_choice_map = {}
        for agent_idx, sub_action in enumerate(final_actions):
            if sub_action >= 3:
                task_idx = sub_action - 3
                agent = self.model.sorted_agents[agent_idx]
                task = self.model.sorted_tasks[task_idx]
                dist = get_distance(agent.pos, task.pos)
                task_choice_map.setdefault(task_idx, []).append((agent_idx, dist))

        for _task_idx, agent_distances in task_choice_map.items():
            if len(agent_distances) > 1:
                winner, _ = min(agent_distances, key=lambda item: item[1])
                for agent_idx, _dist in agent_distances:
                    if agent_idx != winner:
                        final_actions[agent_idx] = 0

        forced_idle = int(sum(1 for action, final in zip(actions, final_actions) if action != final and final == 0))
        num_conflicts = int(sum(1 for choices in task_choice_map.values() if len(choices) > 1))
        unique_assigned = int(len({action - 3 for action in final_actions if action >= 3}))

        reward = 0.0
        for agent, action in zip(self.model.sorted_agents, final_actions):
            _action, indiv_reward = agent.perform_action(action)
            reward += float(indiv_reward)

        self.model.numSteps += 1
        self.t += 1
        terminated = bool(self._is_terminated())

        self.ep_steps += 1
        self.ep_reward += reward
        self.ep_forced_idle += forced_idle
        self.ep_num_conflicts += num_conflicts
        self.ep_J_upper.append(j_upper)
        self.ep_unique_assigned.append(unique_assigned)
        self.ep_deterministic_agents.append(deterministic_agents)

        episode_limit_reached = (self.t >= self.episode_limit) and not all(
            task.completed == 1 for task in self.model.sorted_tasks
        )
        info = {
            "forced_idle": forced_idle,
            "num_conflicts": num_conflicts,
            "J_upper": j_upper,
            "unique_tasks_assigned": unique_assigned,
            "deterministic_agents": deterministic_agents,
            "episode_limit": bool(episode_limit_reached),
        }
        return float(reward), terminated, info

    def _is_terminated(self) -> bool:
        all_completed = all(task.completed == 1 for task in self.model.sorted_tasks)
        timeup = self.t >= self.episode_limit
        return bool(all_completed or timeup)

    def get_obs(self):
        state = np.array(self.model.get_global_state(), dtype=np.float32)
        return np.tile(state[None, :], (self.n_agents, 1))

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return len(self.model.get_global_state())

    def get_state(self):
        return np.array(self.model.get_global_state(), dtype=np.float32)

    def get_state_size(self):
        return len(self.model.get_global_state())

    def get_avail_agent_actions(self, agent_id):
        agent = self.model.sorted_agents[agent_id]
        valid = agent.get_valid_actions()
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for action in valid:
            mask[action] = 1.0
        return mask

    def get_avail_actions(self):
        avail = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        for agent_idx in range(self.n_agents):
            avail[agent_idx] = self.get_avail_agent_actions(agent_idx)
        return avail

    def get_total_actions(self):
        return self.n_actions

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def get_stats(self):
        j_upper_median = float(np.median(self.ep_J_upper)) if self.ep_J_upper else 0.0
        unique_assigned_mean = float(np.mean(self.ep_unique_assigned)) if self.ep_unique_assigned else 0.0
        deterministic_agents_mean = float(np.mean(self.ep_deterministic_agents)) if self.ep_deterministic_agents else 0.0
        return {
            "ForcedIdle_sum": int(self.ep_forced_idle),
            "NumConflicts_sum": int(self.ep_num_conflicts),
            "J_upper_median": float(j_upper_median),
            "UniqueTasksAssigned_mean": float(unique_assigned_mean),
            "DeterministicAgents_mean": float(deterministic_agents_mean),
        }

    def close(self):
        return

    def save_replay(self):
        return
