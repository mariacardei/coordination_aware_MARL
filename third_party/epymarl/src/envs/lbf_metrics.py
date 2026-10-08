from collections import defaultdict
from collections.abc import Iterable
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np
from lbforaging.foraging.environment import Action

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
import envs.pretrained as pretrained  # noqa


class LBFMetricsWrapper(MultiAgentEnv):
    """EPyMARL Gymma-compatible LBF wrapper with process-level diagnostics."""

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

    @property
    def lbf(self):
        return self._env.unwrapped

    def _reset_metrics(self):
        self.ep_steps = 0
        self.initial_food_count = 0
        self.foods_collected = 0
        self.load_events = 0
        self.successful_load_events = 0
        self.failed_load_events = 0
        self.sync_failure_events = 0
        self.insufficient_support_events = 0
        self.movement_opportunities = 0
        self.contested_destinations = 0
        self.contested_movement_agents = 0
        self.movement_actions = 0
        self.noop_actions = 0
        self.successful_load_participation = np.zeros(self.n_agents, dtype=np.int64)
        self.successful_load_partner_sets = [set() for _ in range(self.n_agents)]
        self.excess_loading_capacity_sum = 0
        self.excess_loading_capacity_count = 0
        self.parallel_target_pursuit_sum = 0
        self.parallel_target_pursuit_max = 0
        self.first_feasible_step_by_food = {}
        self.load_sync_delays = []

    def _food_positions(self, field=None):
        field = self.lbf.field if field is None else field
        return {
            (int(r), int(c)): int(field[r, c])
            for r, c in zip(*np.where(field > 0))
        }

    def _adjacent_players_for_pos(self, food_pos, positions=None):
        positions = [p.position for p in self.lbf.players] if positions is None else positions
        fr, fc = food_pos
        return [
            i
            for i, (r, c) in enumerate(positions)
            if (abs(r - fr) == 1 and c == fc) or (abs(c - fc) == 1 and r == fr)
        ]

    def _valid_actions_as_lbf(self, actions):
        valid = []
        for player, a in zip(self.lbf.players, actions):
            try:
                action = Action(int(a))
            except ValueError:
                action = Action.NONE
            if action not in self.lbf._valid_actions[player]:
                action = Action.NONE
            valid.append(action)
        return valid

    @staticmethod
    def _target_position(position, action):
        r, c = position
        if action == Action.NORTH:
            return (r - 1, c)
        if action == Action.SOUTH:
            return (r + 1, c)
        if action == Action.WEST:
            return (r, c - 1)
        if action == Action.EAST:
            return (r, c + 1)
        return position

    def _update_pre_step_metrics(self, actions):
        field_before = np.array(self.lbf.field, copy=True)
        food_before = self._food_positions(field_before)
        positions_before = [p.position for p in self.lbf.players]
        levels = [int(p.level) for p in self.lbf.players]
        valid_actions = self._valid_actions_as_lbf(actions)
        self.noop_actions += sum(1 for action in valid_actions if action == Action.NONE)

        successful_load_candidates = {}
        active_food_targets = set()
        for food_pos, food_level in food_before.items():
            adj = self._adjacent_players_for_pos(food_pos, positions_before)
            if adj:
                active_food_targets.add(food_pos)
            adj_level = sum(levels[i] for i in adj)
            if adj_level >= food_level and food_pos not in self.first_feasible_step_by_food:
                self.first_feasible_step_by_food[food_pos] = self.ep_steps

            loaders = [i for i in adj if valid_actions[i] == Action.LOAD]
            if not loaders:
                continue
            active_food_targets.add(food_pos)
            loading_level = sum(levels[i] for i in loaders)
            self.load_events += 1
            if loading_level >= food_level:
                self.successful_load_events += 1
                successful_load_candidates[food_pos] = (
                    loaders,
                    max(0, loading_level - food_level),
                )
            else:
                self.failed_load_events += 1
                if adj_level >= food_level:
                    self.sync_failure_events += 1
                else:
                    self.insufficient_support_events += 1

        num_parallel_targets = len(active_food_targets)
        self.parallel_target_pursuit_sum += num_parallel_targets
        self.parallel_target_pursuit_max = max(
            self.parallel_target_pursuit_max,
            num_parallel_targets,
        )

        moving = []
        for i, (pos, action) in enumerate(zip(positions_before, valid_actions)):
            if action in {Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST}:
                moving.append((i, self._target_position(pos, action)))
        self.movement_actions += len(moving)
        if moving:
            self.movement_opportunities += 1
        by_dest = defaultdict(list)
        for i, dest in moving:
            by_dest[dest].append(i)
        contested = [agents for agents in by_dest.values() if len(agents) >= 2]
        self.contested_destinations += len(contested)
        self.contested_movement_agents += sum(len(agents) for agents in contested)

        return food_before, successful_load_candidates

    def _update_post_step_metrics(self, food_before, successful_load_candidates):
        food_after = self._food_positions()
        removed = set(food_before) - set(food_after)
        self.foods_collected += len(removed)
        completion_step = self.ep_steps
        for food_pos in removed:
            if food_pos in self.first_feasible_step_by_food:
                self.load_sync_delays.append(
                    completion_step - self.first_feasible_step_by_food[food_pos]
                )
            if food_pos in successful_load_candidates:
                loaders, excess_capacity = successful_load_candidates[food_pos]
                for agent_id in loaders:
                    self.successful_load_participation[agent_id] += 1
                    self.successful_load_partner_sets[agent_id].update(
                        other for other in loaders if other != agent_id
                    )
                self.excess_loading_capacity_sum += excess_capacity
                self.excess_loading_capacity_count += 1

    def step(self, actions):
        actions = [int(a) for a in actions]
        food_before, successful_load_candidates = self._update_pre_step_metrics(actions)
        obs, reward, done, truncated, self._info = self._env.step(actions)
        self.ep_steps += 1
        self._update_post_step_metrics(food_before, successful_load_candidates)
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
        raise self._obs[agent_id]

    def get_obs_size(self):
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        if hasattr(self._env.unwrapped, "state_size"):
            return self._env.unwrapped.state_size
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
        self.initial_food_count = len(self._food_positions())
        self._obs = self._pad_observation(obs)
        return self._obs, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.unwrapped.seed(seed)

    def save_replay(self):
        pass

    @staticmethod
    def _safe_div(num, den):
        return float(num) / float(den) if den else 0.0

    def get_stats(self):
        steps = max(1, int(self.ep_steps))
        avg_delay = float(np.mean(self.load_sync_delays)) if self.load_sync_delays else 0.0
        participation_total = int(self.successful_load_participation.sum())
        if participation_total:
            participation_probs = self.successful_load_participation / participation_total
            participation_entropy = -float(
                np.sum(participation_probs[participation_probs > 0] * np.log(participation_probs[participation_probs > 0]))
            )
            participation_balance = participation_entropy / np.log(self.n_agents) if self.n_agents > 1 else 1.0
            participation_cv = float(np.std(self.successful_load_participation) / np.mean(self.successful_load_participation)) if np.mean(self.successful_load_participation) else 0.0
        else:
            participation_balance = 0.0
            participation_cv = 0.0
        partner_counts = [len(partners) for partners in self.successful_load_partner_sets]
        return {
            "FoodCompletionRatio": self._safe_div(self.foods_collected, self.initial_food_count),
            "FoodsCollected_sum": int(self.foods_collected),
            "FoodsInitial": int(self.initial_food_count),
            "CollectionThroughput": self._safe_div(self.foods_collected, steps),
            "LoadEvents_sum": int(self.load_events),
            "SuccessfulLoadEvents_sum": int(self.successful_load_events),
            "SuccessfulLoadRate": self._safe_div(self.successful_load_events, self.load_events),
            "FailedLoadEvents_sum": int(self.failed_load_events),
            "FailedLoadRate": self._safe_div(self.failed_load_events, self.load_events),
            "SynchronizationFailureEvents_sum": int(self.sync_failure_events),
            "SynchronizationFailureRate": self._safe_div(self.sync_failure_events, self.load_events),
            "InsufficientSupportLoadEvents_sum": int(self.insufficient_support_events),
            "InsufficientSupportLoadRate": self._safe_div(self.insufficient_support_events, self.load_events),
            "MovementContentionDestinations_sum": int(self.contested_destinations),
            "MovementOpportunities_sum": int(self.movement_opportunities),
            "MovementContentionRate_per_step": self._safe_div(self.contested_destinations, steps),
            "MovementContentionRate_per_movement_opportunity": self._safe_div(self.contested_destinations, self.movement_opportunities),
            "MovementContentionAgents_sum": int(self.contested_movement_agents),
            "MovementActions_sum": int(self.movement_actions),
            "IdleOrNoopActions_sum": int(self.noop_actions),
            "IdleOrNoopRate": self._safe_div(self.noop_actions, steps * self.n_agents),
            "MovementContentionAgents_per_movement_action": self._safe_div(self.contested_movement_agents, self.movement_actions),
            "LoadSynchronizationDelay_mean": avg_delay,
            "LoadSynchronizationDelay_count": int(len(self.load_sync_delays)),
            "ParticipationBalance_entropy": participation_balance,
            "ParticipationImbalance_cv": participation_cv,
            "SuccessfulLoadParticipation_sum": participation_total,
            "PartnerDiversity_mean": float(np.mean(partner_counts)) if partner_counts else 0.0,
            "PartnerDiversity_frac": self._safe_div(float(np.mean(partner_counts)) if partner_counts else 0.0, self.n_agents - 1),
            "PartnerDiversity_max": int(max(partner_counts)) if partner_counts else 0,
            "ExcessLoadingCapacity_mean": self._safe_div(self.excess_loading_capacity_sum, self.excess_loading_capacity_count),
            "ExcessLoadingCapacity_sum": int(self.excess_loading_capacity_sum),
            "ParallelTargetPursuit_mean": self._safe_div(self.parallel_target_pursuit_sum, steps),
            "ParallelTargetPursuit_max": int(self.parallel_target_pursuit_max),
            "EpisodeLength": int(self.ep_steps),
        }
