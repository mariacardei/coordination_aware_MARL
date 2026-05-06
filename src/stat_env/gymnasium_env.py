"""Gymnasium integration for STAT."""

from __future__ import annotations

from .config import STATConfig
from .stat_gym import STATGymEnv as BaseSTATGymEnv


class STATGymEnv(BaseSTATGymEnv):
    """Gymnasium-compatible STAT environment."""

    @classmethod
    def from_config(cls, config: STATConfig) -> "STATGymEnv":
        return cls(
            seed=config.seed,
            agents=config.agents,
            tasks=config.tasks,
            width=config.width,
            height=config.height,
            num_bins=config.num_bins,
            agent_speeds=list(config.speeds) if config.speeds else None,
        )
