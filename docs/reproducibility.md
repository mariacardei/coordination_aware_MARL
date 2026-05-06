# Reproducibility Notes

This file maps the paper components to code paths in this release artifact.

## Paper-to-Code Map

| Paper component | Code path |
| --- | --- |
| STAT environment | `src/stat_env/stat_model.py`, `src/stat_env/stat_gym.py` |
| DQN baseline | `src/dqn/train_dqn.py`, `src/dqn/eval_dqn.py` |
| FDQN baseline | `src/fdqn/train_fdqn.py`, `src/fdqn/eval_fdqn.py` |
| IQL, VDN, QMIX, QTRAN, COMA | `third_party/pymarl/src/` |
| PyMARL STAT adapter | `third_party/pymarl/src/envs/stat_env.py` |

## Smoke Tests

Run the STAT environment smoke test:

```bash
python tests/test_stat_env_smoke.py
```

Run a configurable smoke test for any method:

```bash
python scripts/run_smoke.py --method qmix --agents 3 --tasks 6 --width 5 --height 3 --steps 50
```

Supported methods are `stat`, `dqn`, `fdqn`, `iql`, `vdn`, `qmix`, `qtran`, and `coma`. The same command-line flags customize the number of agents, tasks, and grid size for every method.

Smoke-test defaults are also available as YAML files:

```bash
python scripts/run_smoke.py --config configs/smoke/qmix.yaml
python scripts/run_smoke.py --config configs/smoke/qmix.yaml --agents 5 --tasks 12 --width 10 --height 6
```

`configs/smoke/stat.yaml` checks only the STAT environment. `configs/smoke/all.yaml` runs every registered method with tiny settings as a convenience suite.

## Small Training Runs

Joint-action DQN:

```bash
PYTHONPATH=src python src/dqn/train_dqn.py --agents 3 --tasks 6 --width 5 --height 3 --total_steps 200 --batch_size 16 --save_every_steps 100 --eval_max_steps 50 --out_dir results/training
```

Factorized DQN:

```bash
PYTHONPATH=src python src/fdqn/train_fdqn.py --agents 3 --tasks 6 --width 5 --height 3 --total_steps 200 --batch_size 16 --save_every_steps 100 --eval_max_steps 50 --out_dir results/training
```

If `--out_dir` is omitted, both scripts write to `results/training/` at the repository root.

## Full Runs

Full paper runs use the same entry points with larger step budgets, multiple training seeds, and the environment configurations reported in the paper. 

