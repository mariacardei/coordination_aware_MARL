from __future__ import annotations

from .config import STATConfig
from .gymnasium_env import STATGymEnv
from .pymarl_env import STATPyMARLEnv


def make_stat_env(config: STATConfig | None = None, *, integration: str = "gymnasium", **overrides):
    """Create a STAT environment for a supported integration.

    Parameters passed as keyword overrides use the public config names
    (`agents`, `tasks`, `width`, `height`, etc.).
    """
    if config is None:
        config = STATConfig(**overrides)
    elif overrides:
        config = STATConfig(**{**config.__dict__, **overrides})

    normalized = integration.lower().replace("-", "_")
    if normalized in {"gym", "gymnasium"}:
        return STATGymEnv.from_config(config)
    if normalized in {"pymarl", "py_marl"}:
        return STATPyMARLEnv.from_config(config)
    raise ValueError(f"Unsupported STAT integration: {integration!r}.")
