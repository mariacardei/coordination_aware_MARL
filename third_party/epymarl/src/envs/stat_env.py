"""STAT environment adapter for EPyMARL.

This file lives only inside the NeurIPS rebuttal EPyMARL clone. It reuses the
same underlying STAT model/action semantics as the existing PyMARL adapter:
0=idle, 1=move, 2=execute task, 3+i=select task i.
"""

from __future__ import annotations

import random
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
for import_path in (REPO_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from stat_env.core import STATModel, get_distance  # noqa: E402
from .multiagentenv import MultiAgentEnv  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover - torch may not be imported in env workers yet
    torch = None


class STATEnv(MultiAgentEnv):
    def __init__(
        self,
        seed=0,
        n_agents=None,
        n_tasks=None,
        n_responders=None,
        n_victims=None,
        width=10,
        height=6,
        num_bins=5,
        episode_limit=700,
        responder_speeds=None,
        common_reward=True,
        reward_scalarisation="sum",
        key=None,
    ):
        self.seed_value = int(seed)
        self.n_agents = int(n_agents if n_agents is not None else n_responders)
        self.n_tasks = int(n_tasks if n_tasks is not None else n_victims)
        self.width = int(width)
        self.height = int(height)
        self.num_bins = int(num_bins)
        self.episode_limit = int(episode_limit)
        self.common_reward = bool(common_reward)
        self.reward_scalarisation = reward_scalarisation
        self.key = key or f"STAT_{self.n_agents}A{self.n_tasks}T_{self.width}x{self.height}"
        self.n_actions = 3 + self.n_tasks
        self.responder_speeds = responder_speeds or [1.0] * self.n_agents
        self.t = 0
        self._build_model(self.seed_value)
        self._reset_episode_stats()

    def _build_model(self, seed):
        self.model = STATModel(
            seed,
            self.n_agents,
            self.n_tasks,
            self.width,
            self.height,
            policy=6,
            num_bins=self.num_bins,
            agent_speeds=self.responder_speeds,
        )

    def _reset_episode_stats(self):
        self.ep_steps = 0
        self.ep_reward = 0.0
        self.ep_forced_idle = 0
        self.ep_num_conflicts = 0
        self.ep_total_assignments = 0
        self.ep_J_upper = []
        self.ep_unique_assigned = []
        self.ep_deterministic_agents = []

    @staticmethod
    def _safe_div(num, denom):
        denom = float(denom)
        return float(num) / denom if denom > 0 else 0.0

    def seed(self, seed=None):
        if seed is None:
            seed = self.seed_value
        self.seed_value = int(seed)
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        if torch is not None:
            torch.manual_seed(self.seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed_value)
        return [self.seed_value]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            self._build_model(self.seed_value)
        else:
            self.model.reset()
        self.t = 0
        self._reset_episode_stats()
        return self.get_obs(), {}

    def step(self, actions):
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions).reshape(-1)
        actions = [int(x) for x in actions.tolist()]
        final_actions = actions[:]

        avail = self.get_avail_actions()
        valid_per_agent = avail.sum(axis=1).astype(np.int32)
        J_upper = int(np.prod(valid_per_agent)) if valid_per_agent.size > 0 else 0
        deterministic_agents = int((valid_per_agent == 1).sum())

        task_choice_map = {}
        for agent_id, action in enumerate(final_actions):
            if action >= 3:
                task_id = action - 3
                agent = self.model.sorted_agents[agent_id]
                task = self.model.sorted_tasks[task_id]
                dist = get_distance(agent.pos, task.pos)
                task_choice_map.setdefault(task_id, []).append((agent_id, dist))

        for _task_id, choices in task_choice_map.items():
            if len(choices) <= 1:
                continue
            winner, _ = min(choices, key=lambda x: (x[1], x[0]))
            for agent_id, _dist in choices:
                if agent_id != winner:
                    final_actions[agent_id] = 0

        forced_idle = int(sum(1 for a, fa in zip(actions, final_actions) if a != fa and fa == 0))
        num_conflicts = int(sum(1 for choices in task_choice_map.values() if len(choices) > 1))
        unique_assigned = int(len({a - 3 for a in final_actions if a >= 3}))

        reward = 0.0
        for agent, action in zip(self.model.sorted_agents, final_actions):
            _action, indiv_reward = agent.perform_action(action)
            reward += float(indiv_reward)

        self.model.numSteps += 1
        self.t += 1
        terminated = bool(self._is_terminated())
        truncated = False

        self.ep_steps += 1
        self.ep_reward += reward
        self.ep_forced_idle += forced_idle
        self.ep_num_conflicts += num_conflicts
        self.ep_total_assignments += unique_assigned
        self.ep_J_upper.append(J_upper)
        self.ep_unique_assigned.append(unique_assigned)
        self.ep_deterministic_agents.append(deterministic_agents)

        episode_limit_reached = (self.t >= self.episode_limit) and (
            not all(task.completed == 1 for task in self.model.sorted_tasks)
        )
        info = {
            "forced_idle": forced_idle,
            "num_conflicts": num_conflicts,
            "J_upper": J_upper,
            "unique_tasks_assigned": unique_assigned,
            "unique_victims_assigned": unique_assigned,
            "deterministic_agents": deterministic_agents,
            "episode_limit": bool(episode_limit_reached),
        }
        return self.get_obs(), float(reward), terminated, truncated, info

    def _is_terminated(self):
        all_done = all(task.completed == 1 for task in self.model.sorted_tasks)
        timeup = self.t >= self.episode_limit
        return bool(all_done or timeup)

    def get_obs(self):
        state = np.asarray(self.model.get_global_state(), dtype=np.float32)
        return np.tile(state[None, :], (self.n_agents, 1))

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return len(self.model.get_global_state())

    def get_state(self):
        return np.asarray(self.model.get_global_state(), dtype=np.float32)

    def get_state_size(self):
        return len(self.model.get_global_state())

    def get_avail_agent_actions(self, agent_id):
        agent = self.model.sorted_agents[agent_id]
        valid = agent.get_valid_actions()
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for action in valid:
            mask[int(action)] = 1.0
        return mask

    def get_avail_actions(self):
        return np.asarray([self.get_avail_agent_actions(i) for i in range(self.n_agents)], dtype=np.float32)

    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        episode_length = int(self.ep_steps)
        steps_safe = max(1, episode_length)
        J_upper_median = float(np.median(self.ep_J_upper)) if self.ep_J_upper else 0.0
        unique_assigned_mean = float(np.mean(self.ep_unique_assigned)) if self.ep_unique_assigned else 0.0
        deterministic_agents_mean = float(np.mean(self.ep_deterministic_agents)) if self.ep_deterministic_agents else 0.0
        decision_active_agents = max(0.0, float(self.n_agents) - deterministic_agents_mean)
        decision_active_agents_frac = self._safe_div(decision_active_agents, self.n_agents)
        decision_opportunity = decision_active_agents * float(steps_safe)
        completed_tasks = int(sum(task.completed == 1 for task in self.model.sorted_tasks))
        timeout_indicator = int((self.t >= self.episode_limit) and completed_tasks < self.n_tasks)
        assignment_diversity_frac = self._safe_div(unique_assigned_mean, self.n_agents)
        return {
            "Reward": float(self.ep_reward),
            "StepsInEpisode": episode_length,
            "EpisodeLength": episode_length,
            "RewardRate_per_step": self._safe_div(self.ep_reward, steps_safe),
            "ConflictRate_per_step": self._safe_div(self.ep_num_conflicts, steps_safe),
            "ConflictRate_per_task_step": self._safe_div(self.ep_num_conflicts, self.n_tasks * steps_safe),
            "Conflicts_per_task": self._safe_div(self.ep_num_conflicts, self.n_tasks),
            "Conflicts_per_decision_opportunity": self._safe_div(self.ep_num_conflicts, decision_opportunity),
            "ForcedIdleRate_per_step": self._safe_div(self.ep_forced_idle, steps_safe),
            "ForcedIdle_per_task": self._safe_div(self.ep_forced_idle, self.n_tasks),
            "ForcedIdle_per_decision_opportunity": self._safe_div(self.ep_forced_idle, decision_opportunity),
            "DecisionActiveAgents_mean": decision_active_agents,
            "DecisionActiveAgents_frac": decision_active_agents_frac,
            "DecisionOpportunity": decision_opportunity,
            "AssignmentDiversity_frac": assignment_diversity_frac,
            "AssignmentDiversity_per_decision_active_agent": self._safe_div(unique_assigned_mean, decision_active_agents),
            "AssignmentDiversity_per_decision_opportunity": self._safe_div(self.ep_total_assignments, decision_opportunity),
            "TaskThroughput": self._safe_div(completed_tasks, steps_safe),
            "TaskCompletionRatio": self._safe_div(completed_tasks, self.n_tasks),
            "TimeoutIndicator": timeout_indicator,
            "TotalAssignments": int(self.ep_total_assignments),
            "ForcedIdle_sum": int(self.ep_forced_idle),
            "NumConflicts_sum": int(self.ep_num_conflicts),
            "J_upper_median": J_upper_median,
            "UniqueTasksAssigned_mean": unique_assigned_mean,
            "UniqueVictimsAssigned_mean": unique_assigned_mean,
            "DeterministicAgents_mean": deterministic_agents_mean,
        }

    def render(self):
        return None

    def close(self):
        return None

    def save_replay(self):
        return None
