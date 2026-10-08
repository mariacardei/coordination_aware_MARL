# Release Scope

This anonymized artifact contains executable code and configuration examples
for the process-aware MARL experiments in the paper.

## Included

- STAT environment source and Gymnasium/PyMARL/EPyMARL adapters.
- DQN and FDQN training/evaluation entry points.
- Vendored PyMARL code used for value-based STAT baselines, including IQL,
  VDN, QMIX, and QTRAN. COMA support is retained from the framework for
  extensibility but is not part of the main paper comparisons.
- Vendored EPyMARL code used for STAT/LBF/RWARE experiments, including IQL,
  VDN, QMIX, MAPPO, IPPO, and MAA2C configurations.
- LBF and RWARE wrappers with process-diagnostic logging.
- Smoke-test settings and example commands for local executable checks.
- Example configuration files listing STAT, LBF, and RWARE environments used in
  the paper-scale runs.

## Not Included

- Large result directories, trained checkpoints, and cluster log folders. These
  can be regenerated from the released code and are better distributed as a
  separate results archive if needed.
- Hardware-specific launch state. Cluster `sbatch` commands in the paper can be
  reconstructed from the documented local commands and budgets.

## Notes

The top-level smoke tests are intentionally tiny. Passing them verifies that the
code paths are executable; it does not reproduce paper-scale performance. Full
experiments require the budgets, seeds, and environment configurations reported
in the paper.
