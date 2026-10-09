# Coordination-Aware MARL

This anonymized artifact accompanies the paper's STAT, LBF, and RWARE experiments. It
contains the STAT environment, method implementations, experiment
configurations, process-diagnostic logging code, and lightweight smoke tests
for executable verification.

The release is organized so users can first run small local checks, then
scale the same entry points to the full paper configurations.

## Contents

| Component | Location |
| --- | --- |
| STAT environment | `src/stat_env/` |
| DQN and FDQN implementations | `src/dqn/`, `src/fdqn/` |
| Value-based PyMARL implementations | `third_party/pymarl/` |
| EPyMARL implementations and wrappers | `third_party/epymarl/` |
| STAT/LBF/RWARE process metrics | `src/stat_env/`, `third_party/epymarl/src/envs/` |
| Smoke-test configs | `configs/smoke/` |
| Example paper-scale configs/commands | `configs/examples/`, `docs/reproducibility.md` |

The included algorithms are DQN, FDQN, IQL, VDN, QMIX, QTRAN, MAPPO, IPPO,
and MAA2C. The vendored PyMARL tree is retained for the original value-based
STAT baselines, including QTRAN. The vendored EPyMARL tree provides the
policy-gradient methods, LBF/RWARE wrappers, and additional STAT runs.

## Setup

Create an environment with Python 3.10 or 3.11, then install the core
dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r third_party/epymarl/requirements.txt
```

Optional environment packages are needed for LBF and RWARE:

```bash
python -m pip install lbforaging rware
```

If your package index uses newer names or versions, install the package that
provides the Gymnasium IDs used in `configs/examples/lbf_examples.yaml` and
`configs/examples/rware_examples.yaml`.

## Quick Checks

Check installed dependencies first:

```bash
python scripts/check_install.py
```


Run the standalone STAT environment smoke test:

```bash
PYTHONPATH=src python tests/test_stat_env_smoke.py
```

Run a tiny DQN or FDQN training smoke test:

```bash
python scripts/run_smoke.py --config configs/smoke/dqn.yaml
python scripts/run_smoke.py --config configs/smoke/fdqn.yaml
```

Run a tiny PyMARL value-based smoke test on STAT:

```bash
python scripts/run_smoke.py --config configs/smoke/qmix.yaml
```

Run an EPyMARL smoke test on STAT, LBF, or RWARE:

```bash
python scripts/run_epymarl.py --method mappo --env stat --steps 200 --test-nepisode 2
python scripts/run_epymarl.py --method ippo --env lbf --env-id lbforaging:Foraging-8x8-2p-2f-coop-v3 --steps 200
python scripts/run_epymarl.py --method mappo --env rware \
  --env-id rware:rware-tiny-2ag-v2 --common-reward \
  --reward-scalarisation sum --episode-limit 500 --steps 200
```

These are intentionally tiny executable checks, not meaningful training runs.
Use the same scripts with larger `--steps`, multiple seeds, and the
environment IDs listed in `configs/examples/` to reproduce paper-scale runs.
Smoke-test metrics should not be interpreted as paper results.

## Process Diagnostics

STAT logs assignment conflicts, forced idle, assignment diversity, task
throughput, and related normalized quantities. LBF logs failed loads,
successful loads, synchronization failures, collection throughput,
participation balance, and partner diversity. RWARE logs shelf deliveries,
delivery throughput, canceled movements, no-op rate, steps per delivery, and
per-agent delivery contributions.

See `docs/reproducibility.md` for the paper-to-code map and example commands.

## Notes

- Full paper runs are compute intensive and should be launched with the
  training budgets and seeds reported in the paper for reproducible results.
- Smoke tests write temporary outputs under `results/smoke/` unless
  `--keep_outputs` is passed.
- Cluster-specific `sbatch` scripts are examples only; local commands are
  provided for executable verification.

<!-- ## Citation

If you use this code or the STAT environment, please cite:

```bibtex
@article{anonymous2026coordination,
  title={Coordination Matters: Evaluation of Cooperative Multi-Agent Reinforcement Learning},
  author={Anonymous Authors},
  journal={arXiv preprint arXiv:2605.06557},
  year={2026},
  url={https://arxiv.org/abs/2605.06557}
}
``` -->
