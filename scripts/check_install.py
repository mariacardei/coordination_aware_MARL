#!/usr/bin/env python3
"""Check whether the artifact's common runtime dependencies are importable."""

from __future__ import annotations

import importlib
import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
    ("gymnasium", "gymnasium"),
    ("torch", "torch"),
    ("sacred", "sacred"),
]

OPTIONAL_ENVS = [
    ("lbforaging", "lbforaging"),
    ("rware", "rware"),
]


def check(items: list[tuple[str, str]]) -> list[str]:
    missing = []
    for module, package in items:
        try:
            importlib.import_module(module)
            print(f"[ok] {module}")
        except Exception as exc:  # pragma: no cover - diagnostic script
            print(f"[missing] {module} ({package}): {exc}")
            missing.append(package)
    return missing


def main() -> int:
    print("Core dependencies:")
    missing = check(REQUIRED)
    print("\nOptional LBF/RWARE environment packages:")
    optional_missing = check(OPTIONAL_ENVS)

    if missing:
        print("\nInstall core dependencies with:")
        print("  python -m pip install -r requirements.txt")
        print("  python -m pip install -r third_party/epymarl/requirements.txt")
    if optional_missing:
        print("\nInstall optional environment packages for LBF/RWARE with:")
        print("  python -m pip install lbforaging rware")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
