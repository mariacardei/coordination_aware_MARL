# PyMARL and EPyMARL Integration

This artifact includes two related MARL framework trees.

- `third_party/pymarl/` contains the compact PyMARL-style code used for the
  original value-based STAT baselines and QTRAN support.
- `third_party/epymarl/` contains the EPyMARL-style code used for MAPPO, IPPO,
  MAA2C, and the LBF/RWARE diagnostic wrappers.

The top-level scripts hide most framework-specific path setup:

```bash
python scripts/run_smoke.py --config configs/smoke/qmix.yaml
python scripts/run_epymarl.py --method mappo --env stat --steps 200
```

When running framework entry points directly, set `PYTHONPATH` to include both
`src` and the relevant framework `src` directory.
