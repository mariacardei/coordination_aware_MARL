from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class STATConfig:
    """Configuration shared by all STAT environment integrations."""

    agents: int = 3
    tasks: int = 6
    width: int = 5
    height: int = 3
    num_bins: int = 5
    seed: int = 0
    episode_limit: int = 700
    agent_speeds: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("agents", "tasks", "width", "height", "num_bins", "episode_limit"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.agent_speeds is not None and len(self.agent_speeds) != self.agents:
            raise ValueError(
                "agent_speeds must have one entry per agent "
                f"({self.agents}), got {len(self.agent_speeds)}."
            )

    @classmethod
    def from_kwargs(cls, **kwargs) -> "STATConfig":
        """Build a config from keyword arguments."""
        defaults = cls()
        return cls(
            agents=int(kwargs.pop("agents", defaults.agents)),
            tasks=int(kwargs.pop("tasks", defaults.tasks)),
            width=int(kwargs.pop("width", defaults.width)),
            height=int(kwargs.pop("height", defaults.height)),
            num_bins=int(kwargs.pop("num_bins", defaults.num_bins)),
            seed=int(kwargs.pop("seed", defaults.seed)),
            episode_limit=int(kwargs.pop("episode_limit", defaults.episode_limit)),
            agent_speeds=kwargs.pop("agent_speeds", None),
        )

    @property
    def speeds(self) -> tuple[float, ...] | None:
        return self.agent_speeds

    @property
    def per_agent_actions(self) -> int:
        return 3 + self.tasks

    @property
    def dqn_joint_actions(self) -> int:
        return self.per_agent_actions ** self.agents
