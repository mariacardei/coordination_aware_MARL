#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import sys

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
pymarl_main = repo_root / "third_party" / "pymarl" / "src" / "main.py"
results_dir = repo_root / "results" / "smoke" / "coma"

cmd = [
    sys.executable, str(pymarl_main),
    "--config=coma",
    "--env-config=stat",
    "with",
    "use_cuda=False",
    f"local_results_path={results_dir}",
    "name=COMA_DEBUG",
    "label=smoke_test",
    "env_args.agents=3",
    "env_args.tasks=6",
    "env_args.width=5",
    "env_args.height=3",
    "env_args.num_bins=5",
    "env_args.episode_limit=50",
    "seed=0",
    "t_max=50",
    "test_interval=25",
    "test_nepisode=1",
    "runner_log_interval=100",
    "save_model=False",
    "use_tensorboard=False",
    "buffer_cpu_only=True",
]

env = os.environ.copy()
env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root / 'third_party' / 'pymarl' / 'src'}"

print("Running COMA smoke test:\n")
print(" ".join(cmd), "\n")

result = subprocess.run(cmd, env=env)

if result.returncode == 0:
    print("\nCOMA smoke test finished successfully.")
else:
    print(f"\nCOMA smoke test failed with exit code {result.returncode}.")
    sys.exit(result.returncode)
