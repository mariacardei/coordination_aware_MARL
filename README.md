# Coordination-Aware MARL Evaluation

This repository contains the release artifact for the paper "Coordination Matters: Evaluation of Cooperative Multi-Agent Reinforcement Learning".

The code implements STAT, a commitment-constrained spatial task-allocation testbed, and the MARL baselines used to evaluate coordination-aware diagnostics: DQN, FDQN, IQL, VDN, QMIX, QTRAN, and COMA.

This release contains source code, training/evaluation entry points, PyMARL integration, and documentation. 

## Repository Layout

```text
src/
  stat_env/      STAT model and Gymnasium wrapper
  dqn/           Joint-action DQN training and evaluation
  fdqn/          Factorized DQN training and evaluation
configs/
  smoke/         Small runnable configs for executable checks
third_party/
  pymarl/        PyMARL-based IQL, VDN, QMIX, QTRAN, and COMA code
tests/
  smoke tests for environment and PyMARL execution
docs/
  integration, extension, and run notes
results/
  generated outputs; ignored by git except placeholders
```

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You can also use `pip install -r requirements.txt` when you do not want an editable package install, but `pip install -e .` is the recommended path.

## Running STAT

STAT has one underlying environment model and two wrappers:

- `STATGymEnv` is the Gymnasium-style wrapper used by DQN, FDQN, standalone scripts, and most new methods.
- `STATPyMARLEnv` is an adapter for PyMARL's runner API. It exposes the same STAT dynamics through PyMARL methods such as IQL, VDN, QMIX, QTRAN, and COMA.

Create STAT directly from Python:

```python
from stat_env import STATConfig, make_stat_env

env = make_stat_env(
    STATConfig(agents=3, tasks=6, width=5, height=3),
    integration="gymnasium",
)
```

The same config can create the PyMARL-compatible wrapper:

```python
from stat_env import STATConfig, make_stat_env

pymarl_env = make_stat_env(
    STATConfig(agents=3, tasks=6, width=5, height=3),
    integration="pymarl",
)
```

## Quick Executable Checks

The `scripts/run_smoke.py` entry point is a smoke runner: it uses tiny step budgets to check that a method launches, interacts with STAT, and writes outputs. These runs are not large-scale experiments and should not be used to interpret final performance.

How to run a quick check for one method:

```bash
python scripts/run_smoke.py --method qmix --agents 3 --tasks 6 --width 5 --height 3 --steps 50
```

Supported methods are `stat`, `dqn`, `fdqn`, `iql`, `vdn`, `qmix`, `qtran`, and `coma`. Use `--method all` to run the full smoke suite.
Smoke-test outputs are written under `results/smoke/` and removed after successful runs unless `--keep_outputs` is set.

You can also start from a YAML config and override fields on the command line

```bash
python scripts/run_smoke.py --config configs/smoke/qmix.yaml
python scripts/run_smoke.py --config configs/smoke/qmix.yaml --agents 5 --tasks 12 --width 10 --height 6
```

The smoke configs are convenience defaults:

- `configs/smoke/stat.yaml` checks only the STAT environment with random valid actions.
- `configs/smoke/all.yaml` runs every registered method with very small settings.
- Method configs such as `qmix.yaml` or `dqn.yaml` run one method and can be overridden from the command line.

Examples:

```bash
python scripts/run_smoke.py --method dqn --agents 2 --tasks 4 --width 6 --height 4 --steps 100
python scripts/run_smoke.py --method coma --agents 5 --tasks 12 --width 10 --height 6 --steps 50
```

## Training Runs

For direct DQN or FDQN runs, use `--out_dir` to keep generated artifacts under a selected directory:

```bash
PYTHONPATH=src python src/dqn/train_dqn.py --agents 3 --tasks 6 --width 5 --height 3 --total_steps 200 --out_dir results/training
PYTHONPATH=src python src/fdqn/train_fdqn.py --agents 3 --tasks 6 --width 5 --height 3 --total_steps 200 --out_dir results/training
```

For PyMARL methods, call the vendored PyMARL entry point with `stat` as the environment config. This uses the PyMARL wrapper around the same STAT environment:

```bash
PYTHONPATH=src:third_party/pymarl/src python third_party/pymarl/src/main.py \
  --config=qmix --env-config=stat with \
  env_args.agents=3 env_args.tasks=6 env_args.width=5 env_args.height=3 \
  t_max=1000 local_results_path=results/training/qmix use_cuda=False
```

Replace `qmix` with `iql`, `vdn`, `qtran`, or `coma` to run another PyMARL method.

## Adding Methods

New methods can use STAT through either the Gymnasium-style wrapper or the PyMARL wrapper. To make a new method runnable through the shared smoke runner, add a small launcher in `scripts/run_smoke.py`, register the method name in `METHODS`, and add a config under `configs/smoke/`.

See [docs/adding_methods.md](docs/adding_methods.md) for the expected integration points.

## Notes

- Generated outputs go under `results/`.
- PyMARL integration details are in [docs/pymarl_integration.md](docs/pymarl_integration.md), and third-party code notices are in `THIRD_PARTY_NOTICES.md`.
- This project is released under the MIT License; see `LICENSE`. Vendored PyMARL keeps its own Apache-2.0 license under `third_party/pymarl/LICENSE`.
