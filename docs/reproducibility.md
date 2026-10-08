# Reproducibility Notes

This document maps paper components to code paths and gives executable command
templates. Commands are written relative to the repository root.

## Paper-to-Code Map

| Paper component | Code path |
| --- | --- |
| STAT environment | `src/stat_env/` |
| STAT Gymnasium adapter | `src/stat_env/gymnasium_env.py`, `src/stat_env/stat_gym.py` |
| DQN baseline | `src/dqn/train_dqn.py`, `src/dqn/eval_dqn.py` |
| FDQN baseline | `src/fdqn/train_fdqn.py`, `src/fdqn/eval_fdqn.py` |
| PyMARL value baselines | `third_party/pymarl/src/` |
| EPyMARL methods | `third_party/epymarl/src/config/algs/` |
| EPyMARL STAT adapter | `third_party/epymarl/src/envs/stat_env.py` |
| LBF diagnostics wrapper | `third_party/epymarl/src/envs/lbf_metrics.py` |
| RWARE diagnostics wrapper | `third_party/epymarl/src/envs/rware_metrics.py` |
| STAT configurations | `configs/examples/stat_configs.yaml` |
| LBF configurations | `configs/examples/lbf_examples.yaml` |
| RWARE configurations | `configs/examples/rware_examples.yaml` |

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -r third_party/epymarl/requirements.txt
```

For LBF and RWARE experiments, also install the environment packages providing
the Gymnasium IDs listed in `configs/examples/`:

```bash
python -m pip install lbforaging rware
```

## Smoke Tests

Check installed dependencies first:

```bash
python scripts/check_install.py
```


STAT environment only:

```bash
PYTHONPATH=src python tests/test_stat_env_smoke.py
```

DQN/FDQN tiny runs:

```bash
python scripts/run_smoke.py --config configs/smoke/dqn.yaml
python scripts/run_smoke.py --config configs/smoke/fdqn.yaml
```

PyMARL value-based STAT smoke tests:

```bash
python scripts/run_smoke.py --config configs/smoke/iql.yaml
python scripts/run_smoke.py --config configs/smoke/vdn.yaml
python scripts/run_smoke.py --config configs/smoke/qmix.yaml
python scripts/run_smoke.py --config configs/smoke/qtran.yaml
```

EPyMARL STAT smoke tests for policy-gradient methods:

```bash
python scripts/run_epymarl.py --method mappo --env stat --steps 200 --test-nepisode 2
python scripts/run_epymarl.py --method ippo --env stat --steps 200 --test-nepisode 2
python scripts/run_epymarl.py --method maa2c --env stat --steps 200 --test-nepisode 2
```

LBF and RWARE executable checks:

```bash
python scripts/run_epymarl.py --method mappo --env lbf \
  --env-id lbforaging:Foraging-8x8-2p-2f-coop-v3 --steps 200

python scripts/run_epymarl.py --method ippo --env rware \
  --env-id rware:rware-tiny-2ag-v2 --common-reward \
  --reward-scalarisation sum --episode-limit 500 --steps 200
```

## Paper-Scale Templates

Full experiments use the same entry points with larger `--steps`, multiple
seeds, and the environment IDs in `configs/examples/`. For RWARE team-reward
runs, use:

```bash
python scripts/run_epymarl.py --method mappo --env rware \
  --env-id rware:rware-tiny-2ag-v2 --common-reward \
  --reward-scalarisation sum --episode-limit 500 --steps 40000000 \
  --test-nepisode 100 --seed 0
```

Repeat over methods, seeds, and environments as described in the paper. For
STAT DQN/FDQN, use `src/dqn/train_dqn.py` and `src/fdqn/train_fdqn.py`; for
STAT EPyMARL methods, use `scripts/run_epymarl.py --env stat`.

## Diagnostics

Diagnostic metrics are emitted by the environment wrappers through the standard
training/evaluation logs. Key metrics include:

- STAT: assignment conflicts, forced idle, assignment diversity, task
  throughput, and episode length.
- LBF: failed-load rate, successful loads, synchronization failures, collection
  throughput, participation balance, and partner diversity.
- RWARE: shelf deliveries, delivery throughput, canceled movement rate, no-op
  rate, steps per delivery, and per-agent delivery contribution.
