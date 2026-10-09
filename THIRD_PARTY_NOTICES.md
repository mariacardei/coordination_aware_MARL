# Third-Party Notices

This artifact includes local copies of PyMARL and EPyMARL under `third_party/`.
These vendored framework trees are credited to their original authors and remain
governed by their upstream license files. The repository-level license applies
to this artifact's original code, including the STAT environment, integration
adapters, experiment wrappers, and process-diagnostic logging additions.

## PyMARL

PyMARL is included under `third_party/pymarl/` and retains its upstream license
file at `third_party/pymarl/LICENSE`.

The STAT environment adapter in `third_party/pymarl/src/envs/stat_env.py` and
related configuration files are part of this artifact's integration code for
running the STAT environment with PyMARL-style value-based methods.

PyMARL citation: M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar,
N. Nardelli, T. G. J. Rudner, C.-M. Hung, P. H. S. Torr, J. Foerster, and
S. Whiteson. The StarCraft Multi-Agent Challenge. CoRR abs/1902.04043, 2019.

## EPyMARL

EPyMARL is included under `third_party/epymarl/` and retains its upstream
license and notice files at `third_party/epymarl/LICENSE` and
`third_party/epymarl/NOTICE`.

The STAT, LBF, and RWARE diagnostic environment wrappers and related
configuration files in the vendored EPyMARL tree are part of this artifact's
integration code for running the paper's EPyMARL-based experiments.

EPyMARL citation: G. Papoudakis, F. Christianos, L. Schäfer, and S. V. Albrecht.
Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative
Tasks. Proceedings of the Neural Information Processing Systems Track on
Datasets and Benchmarks, 2021.

See `docs/pymarl_integration.md` for the integration boundary between STAT,
PyMARL, and EPyMARL.
