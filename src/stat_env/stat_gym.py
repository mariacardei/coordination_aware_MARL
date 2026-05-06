"""Gymnasium integration for STAT.

STAT is a domain-general spatial task-allocation environment. A team of
agents selects spatially distributed tasks, commits to assignments, moves
toward selected tasks, and completes them after a short service time.

Use `agents`, `tasks`, and `policy` to configure the environment.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .stat_model import DEBUG, STATModel, get_distance


class STATGymEnv(gym.Env):
    """Gymnasium-compatible STAT environment.

    Observation:
        Dict with a global state vector (`obs`) and per-agent action masks
        (`agent_masks`).

    Actions:
        A joint action represented as one sub-action per agent:
        - 0: idle
        - 1: continue moving to current task
        - 2: execute current task
        - 3 + i: select task i
    """

    metadata = {"render_modes": ["human"], "render.modes": ["human"]}

    def __init__(
        self,
        seed: int = 0,
        agents: int = 1,
        tasks: int = 1,
        width: int = 5,
        height: int = 5,
        policy: int = 6,
        num_bins: int = 5,
        agent_speeds=None,
    ):
        super().__init__()

        self.seed_val = seed
        self.num_agents = int(agents)
        self.num_tasks = int(tasks)
        self.width = width
        self.height = height
        self.policy = policy
        self.num_bins = num_bins
        self.episode_reward = 0.0
        self.agent_speeds = agent_speeds or [1.0] * self.num_agents

        self.model = STATModel(
            seed,
            self.num_agents,
            self.num_tasks,
            width,
            height,
            policy,
            num_bins,
            agent_speeds=self.agent_speeds,
        )

        # DQN/FDQN choose joint actions externally, so the environment exposes
        # masks instead of a single Gymnasium Discrete action space.
        self.action_space = None

        sample_state = self.model.get_global_state()
        obs_dim = len(sample_state)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0, high=(self.model.num_bins + 1), shape=(obs_dim,), dtype=np.float32),
                "agent_masks": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.model.num_agents, 3 + self.model.num_tasks),
                    dtype=np.float32,
                ),
            }
        )

    def get_agent_action_masks(self) -> np.ndarray:
        """Return a binary valid-action mask for each agent."""
        masks = []
        for agent in self.model.sorted_agents:
            mask = np.zeros(self.model.num_sub_actions, dtype=np.float32)
            for action in agent.get_valid_actions():
                mask[action] = 1.0
            masks.append(mask)
        return np.array(masks, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
            self.model.reset(seed=seed)
        else:
            self.model.reset()

        self.episode_reward = 0.0
        state = np.array(self.model.get_global_state(), dtype=np.float32)
        return {"obs": state, "agent_masks": self.get_agent_action_masks()}, {}

    def step(self, joint_action):
        """Apply one joint action and return Gymnasium step outputs."""
        final_actions = [int(action) for action in joint_action]

        task_choice_map: dict[int, list[tuple[int, float]]] = {}
        for agent_idx, sub_action in enumerate(final_actions):
            if sub_action >= 3:
                task_idx = sub_action - 3
                agent = self.model.sorted_agents[agent_idx]
                task = self.model.sorted_tasks[task_idx]
                distance = get_distance(agent.pos, task.pos)
                task_choice_map.setdefault(task_idx, []).append((agent_idx, distance))

        for _task_idx, agent_distances in task_choice_map.items():
            if len(agent_distances) > 1:
                winner, _distance = min(agent_distances, key=lambda item: item[1])
                for agent_idx, _distance in agent_distances:
                    if agent_idx != winner:
                        final_actions[agent_idx] = 0

        step_reward = 0.0
        for agent, sub_action in zip(self.model.sorted_agents, final_actions):
            _actual_action, reward = agent.perform_action(sub_action)
            if DEBUG:
                print(
                    f"[DEBUG] Agent {agent.unique_id} pos={agent.pos} "
                    f"final_action={sub_action} reward={reward}"
                )
            step_reward += float(reward)

        self.model.numSteps += 1
        done = all(task.completed == 1 for task in self.model.sorted_tasks)

        state = np.array(self.model.get_global_state(), dtype=np.float32)
        agent_masks = self.get_agent_action_masks()
        valid_per_agent = agent_masks.sum(axis=1).astype(np.int32)
        unique_assignments = int(len({action - 3 for action in final_actions if action >= 3}))

        info = {
            "valid_per_agent": valid_per_agent,
            "J_upper": int(valid_per_agent.prod()),
            "forced_idle": int(sum(1 for action, final in zip(joint_action, final_actions) if action != final and final == 0)),
            "unique_tasks_assigned": unique_assignments,
            "deterministic_agents": int((valid_per_agent == 1).sum()),
            "num_conflicts": int(sum(1 for choices in task_choice_map.values() if len(choices) > 1)),
        }

        return {"obs": state, "agent_masks": agent_masks}, step_reward, done, False, info

    def render(self, mode="human"):
        print(f"Step: {self.model.numSteps}, Total tasks completed: {self.model.total_tasks_completed}")

    def close(self):
        return


try:
    gym.envs.registration.register(id="STAT-v0", entry_point="stat_env.stat_gym:STATGymEnv")
except gym.error.Error:
    pass
