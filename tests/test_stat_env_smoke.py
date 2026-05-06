#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from stat_env import STATConfig, make_stat_env


def main():
    env = make_stat_env(STATConfig(seed=0, agents=3, tasks=6, width=5, height=3, num_bins=5))
    obs, info = env.reset(seed=0)

    assert "obs" in obs
    assert "agent_masks" in obs
    assert obs["agent_masks"].shape == (3, 9)
    assert np.all((obs["agent_masks"] == 0) | (obs["agent_masks"] == 1))

    action = []
    for mask in obs["agent_masks"]:
        valid = np.flatnonzero(mask)
        assert len(valid) > 0
        action.append(int(valid[0]))

    next_obs, reward, done, truncated, step_info = env.step(action)

    assert "obs" in next_obs
    assert "agent_masks" in next_obs
    assert isinstance(float(reward), float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    for key in ["forced_idle", "num_conflicts", "unique_tasks_assigned", "J_upper"]:
        assert key in step_info

    print("STAT environment smoke test passed.")


if __name__ == "__main__":
    main()
