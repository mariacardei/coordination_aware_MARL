"""PyMARL registry shim for STAT.

The environment implementation lives in `stat_env.pymarl_env` so STAT owns its
integrations instead of hiding them inside vendored PyMARL code.
"""

from stat_env.pymarl_env import STATPyMARLEnv

__all__ = ["STATPyMARLEnv"]
