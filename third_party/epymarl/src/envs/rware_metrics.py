from collections.abc import Iterable
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np
import rware  # noqa: F401
from rware.warehouse import Action

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
import envs.pretrained as pretrained  # noqa


class RWAREMetricsWrapper(MultiAgentEnv):
    """EPyMARL Gymma-compatible RWARE wrapper with process diagnostics."""

    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        self._env = gym.make(f"{key}", **kwargs)
        self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.unwrapped.n_agents
        self.episode_limit = time_limit
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except Exception:
            self._env.reset(seed=self._seed)

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation}"
                )
        self._reset_metrics()

    @property
    def rware(self):
        return self._env.unwrapped

    def _reset_metrics(self):
        self.ep_steps = 0
        self.deliveries = 0
        self.agent_deliveries = np.zeros(self.n_agents, dtype=np.int64)
        self.forward_requests = 0
        self.canceled_movements = 0
        self.noop_actions = 0
        self.toggle_load_actions = 0

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def _requested_forward_targets(self, actions):
        targets = []
        for agent, action in zip(self.rware.agents, actions):
            action = int(action)
            if action == Action.NOOP.value:
                self.noop_actions += 1
            elif action == Action.TOGGLE_LOAD.value:
                self.toggle_load_actions += 1

            if action == Action.FORWARD.value:
                self.forward_requests += 1
                start = (agent.x, agent.y)
                target = agent.req_location(self.rware.grid_size)
                targets.append((agent.id - 1, start, target))
        return targets

    def _update_movement_metrics(self, requested_targets):
        for agent_id, start, target in requested_targets:
            agent = self.rware.agents[agent_id]
            end = (agent.x, agent.y)
            if target != start and end == start:
                self.canceled_movements += 1

    def _update_delivery_metrics(self, rewards):
        delivered_agents = [
            idx for idx, reward in enumerate(rewards) if float(reward) > 0.0
        ]
        if not delivered_agents:
            return
        self.deliveries += len(delivered_agents)
        for agent_id in delivered_agents:
            self.agent_deliveries[agent_id] += 1

    def step(self, actions):
        actions = [int(a) for a in actions]
        requested_targets = self._requested_forward_targets(actions)
        obs, reward, done, truncated, self._info = self._env.step(actions)
        self.ep_steps += 1

        rewards_for_metrics = reward if isinstance(reward, Iterable) else [reward]
        self._update_delivery_metrics(rewards_for_metrics)
        self._update_movement_metrics(requested_targets)
        self._obs = self._pad_observation(obs)

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment"
            )

        if isinstance(done, Iterable):
            done = all(done)
        return self._obs, reward, done, truncated, self._info

    def get_obs(self):
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        if hasattr(self.rware, "state_size"):
            return self.rware.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        self._reset_metrics()
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(obs)
        return self._obs, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self.rware.seed(seed)

    def save_replay(self):
        pass

    @staticmethod
    def _safe_div(num, den):
        return float(num) / float(den) if den else 0.0

    def get_stats(self):
        steps = max(1, int(self.ep_steps))
        agent_delivery_mean = (
            float(np.mean(self.agent_deliveries)) if len(self.agent_deliveries) else 0.0
        )
        agent_delivery_cv = (
            float(np.std(self.agent_deliveries) / agent_delivery_mean)
            if agent_delivery_mean
            else 0.0
        )
        stats = {
            "ShelvesDelivered_sum": int(self.deliveries),
            "DeliveryThroughput": self._safe_div(self.deliveries, steps),
            "StepsPerDelivery": self._safe_div(steps, self.deliveries),
            "EpisodeLength": int(self.ep_steps),
            "MovementRequests_sum": int(self.forward_requests),
            "CanceledMovements_sum": int(self.canceled_movements),
            "CanceledMovementRate": self._safe_div(
                self.canceled_movements, self.forward_requests
            ),
            "NoopActions_sum": int(self.noop_actions),
            "NoopRate": self._safe_div(self.noop_actions, steps * self.n_agents),
            "ToggleLoadActions_sum": int(self.toggle_load_actions),
            "AgentDeliveryContribution_mean": agent_delivery_mean,
            "AgentDeliveryContribution_cv": agent_delivery_cv,
        }
        for agent_id, deliveries in enumerate(self.agent_deliveries):
            stats[f"agent_{agent_id}_deliveries"] = int(deliveries)
        return stats
