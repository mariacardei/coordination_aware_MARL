# Adding Methods

You can test new methods in STAT through two supported interfaces. They share the same underlying dynamics and differ only in the methods they expose to training code:

- Gymnasium-style wrapper: `stat_env.make_stat_env(..., integration="gymnasium")`. This is the default choice for a new method unless the method is implemented inside PyMARL.
- PyMARL-compatible wrapper: `stat_env.make_stat_env(..., integration="pymarl")`. This exists so vendored PyMARL algorithms can call STAT through PyMARL's expected environment API.

## Gymnasium-Style Methods

Create an environment from a `STATConfig`:

```python
from stat_env import STATConfig, make_stat_env

env = make_stat_env(
    STATConfig(agents=3, tasks=6, width=5, height=3, episode_limit=50),
    integration="gymnasium",
)
obs, info = env.reset(seed=0)
```

The observation is a dictionary with:

- `obs`: global state vector
- `agent_masks`: valid action mask for each agent

Each agent uses the same action convention:

- `0`: idle
- `1`: move
- `2`: execute task
- `3 + i`: select task `i`

## PyMARL Methods

For methods implemented inside the vendored PyMARL tree, use the STAT env config. This route is mainly for IQL, VDN, QMIX, QTRAN, COMA, or any new method added to the PyMARL config system:

```bash
PYTHONPATH=src:third_party/pymarl/src python third_party/pymarl/src/main.py \
  --config=qmix --env-config=stat with \
  env_args.agents=3 env_args.tasks=6 env_args.width=5 env_args.height=3 \
  t_max=1000 local_results_path=results/training/qmix use_cuda=False
```

To add a new PyMARL algorithm, add its algorithm YAML under `third_party/pymarl/src/config/algs/`, then run it with `--config=<new_name> --env-config=stat`.

## Shared Smoke Runner

To make a new method available through `scripts/run_smoke.py`:

1. Add the method name to `METHODS`.
2. Add a launcher branch in `run_method()`.
3. Add a tiny default config under `configs/smoke/<method>.yaml`.
4. Verify with:

```bash
python scripts/run_smoke.py --config configs/smoke/<method>.yaml
```

Note: smoke runs are executable checks, not benchmark-scale experiments.
