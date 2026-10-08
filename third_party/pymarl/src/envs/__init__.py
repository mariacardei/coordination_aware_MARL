from functools import partial
import sys
import os

REGISTRY = {}

def env_fn(env, **kwargs):
    return env(**kwargs)

# ---- Optional: SMAC / SC2 ----
try:
    from smac.env import MultiAgentEnv, StarCraft2Env
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

    if sys.platform == "linux":
        os.environ.setdefault(
            "SC2PATH",
            os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
        )
except ModuleNotFoundError:
    # SMAC not installed; that is fine for STAT-only runs.
    pass

# ---- Any custom env ----
from .stat_env import STATPyMARLEnv
REGISTRY["stat"] = partial(env_fn, env=STATPyMARLEnv)
