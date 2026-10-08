# Adding Methods

You can test new methods in STAT through three supported interfaces. They share
the same underlying dynamics and differ only in the training API they expose.

- Gymnasium-style wrapper: `stat_env.make_stat_env(..., integration="gymnasium")`.
  This is the default for new standalone methods.
- PyMARL-compatible wrapper: `stat_env.make_stat_env(..., integration="pymarl")`.
  This supports the compact vendored PyMARL value-based baselines.
- EPyMARL environment adapters: `third_party/epymarl/src/envs/stat_env.py`,
  `lbf_metrics.py`, and `rware_metrics.py`. This is the preferred route for
  MAPPO, IPPO, MAA2C, and EPyMARL-based environment variants.

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

For methods implemented inside the vendored PyMARL tree, use the STAT env
config. This route is mainly for IQL, VDN, QMIX, QTRAN, COMA, or new methods
added to the PyMARL config system:

```bash
PYTHONPATH=src:third_party/pymarl/src python third_party/pymarl/src/main.py \
  --config=qmix --env-config=stat with \
  env_args.agents=3 env_args.tasks=6 env_args.width=5 env_args.height=3 \
  t_max=1000 local_results_path=results/training/qmix use_cuda=False
```

To add a new PyMARL algorithm, add its algorithm YAML under
`third_party/pymarl/src/config/algs/`, then run it with
`--config=<new_name> --env-config=stat`.

## EPyMARL Methods and Environments

EPyMARL method configs live under `third_party/epymarl/src/config/algs/`.
Environment configs live under `third_party/epymarl/src/config/envs/`. Use the
top-level wrapper for smoke tests and small runs:

```bash
python scripts/run_epymarl.py --method mappo --env stat --steps 200
python scripts/run_epymarl.py --method ippo --env lbf \
  --env-id lbforaging:Foraging-8x8-2p-2f-coop-v3 --steps 200
python scripts/run_epymarl.py --method maa2c --env rware \
  --env-id rware:rware-tiny-2ag-v2 --common-reward --steps 200
```

To add a new EPyMARL method, add an algorithm YAML in
`third_party/epymarl/src/config/algs/` and, if needed, register learners,
controllers, or modules in the corresponding EPyMARL registries. To add a new
environment diagnostic wrapper, add it under `third_party/epymarl/src/envs/`
and register it in `third_party/epymarl/src/envs/__init__.py`.

## Shared Smoke Runners

For `scripts/run_smoke.py`:

1. Add the method name to `METHODS`.
2. Add a launcher branch in `run_method()`.
3. Add a tiny default config under `configs/smoke/<method>.yaml`.
4. Verify with `python scripts/run_smoke.py --config configs/smoke/<method>.yaml`.

For `scripts/run_epymarl.py`, add the method to `METHODS` if it has an EPyMARL
YAML config and uses one of the supported environment adapters.

Smoke runs are executable checks, not benchmark-scale experiments.
