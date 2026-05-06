import os
import glob
import argparse
import numpy as np
import pandas as pd

from eval_dqn import run_eval


ID_COLS = {"config", "train_seed", "eval_seed", "episode_idx"}

def infer_metrics_from_csv(df):
    # everything except ID cols; only keep numeric-able columns
    metrics = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        # treat as metric if it can be numeric
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            metrics.append(c)
    return metrics

def summarize_config(dqn_root: str, config: str):
    cfg = config if config.startswith("DQN_") else f"DQN_{config}"
    cfg_dir = os.path.join(dqn_root, "evaluation", cfg)
    if not os.path.isdir(cfg_dir):
        print(f"[AGG:SKIP] No eval folder found for {cfg_dir}")
        return

    csvs = sorted(glob.glob(os.path.join(cfg_dir, "eval_results_seed*.csv")))
    if not csvs:
        print(f"[AGG:SKIP] No eval_results_seed*.csv found in {cfg_dir}")
        return

    # Use first CSV to infer metrics (assumes consistent columns across seed files)
    df0 = pd.read_csv(csvs[0])
    metrics = infer_metrics_from_csv(df0)

    # ---- Per-model summaries ----
    model_rows = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)

        # coerce metric columns to numeric
        for m in metrics:
            df[m] = pd.to_numeric(df[m], errors="coerce")

        train_seed = df["train_seed"].iloc[0] if "train_seed" in df.columns else os.path.basename(csv_path)
        n_episodes = len(df)
        n_eval_seeds = df["eval_seed"].nunique() if "eval_seed" in df.columns else np.nan

        row = {
            "config": config,
            "train_seed": train_seed,
            "n_episodes": int(n_episodes),
            "n_eval_seeds": int(n_eval_seeds) if not np.isnan(n_eval_seeds) else np.nan,
        }

        for m in metrics:
            vals = df[m].to_numpy()
            vals = vals[np.isfinite(vals)]
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{m}_std"]  = float(np.std(vals, ddof=0)) if len(vals) else np.nan
            row[f"{m}_min"]  = float(np.min(vals)) if len(vals) else np.nan
            row[f"{m}_max"]  = float(np.max(vals)) if len(vals) else np.nan

        model_rows.append(row)

    model_summary_df = pd.DataFrame(model_rows).sort_values("train_seed")
    model_summary_path = os.path.join(cfg_dir, "model_summary.csv")
    model_summary_df.to_csv(model_summary_path, index=False)

    # ---- Across-model summaries ----
    across = {"config": config, "n_models": int(len(model_summary_df))}
    for m in metrics:
        col = f"{m}_mean"
        if col not in model_summary_df.columns:
            continue
        vals = pd.to_numeric(model_summary_df[col], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        across[f"{m}_across_models_mean"] = float(np.mean(vals))
        across[f"{m}_across_models_std"]  = float(np.std(vals, ddof=0))
        across[f"{m}_across_models_se"]   = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        across[f"{m}_across_models_min"]  = float(np.min(vals))
        across[f"{m}_across_models_max"]  = float(np.max(vals))

    across_df = pd.DataFrame([across])
    across_path = os.path.join(cfg_dir, "config_summary_across_models.csv")
    across_df.to_csv(across_path, index=False)

    print(f"[AGG] Metrics inferred: {metrics}")
    print(f"[AGG] Saved: {model_summary_path}")
    print(f"[AGG] Saved: {across_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, nargs="+", required=True,
                        help="Configs like 3R_5V_5x5 3R_10V_10x10 ...")
    parser.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--eval_seed_start", type=int, default=100)
    parser.add_argument("--eval_seed_end", type=int, default=119)  # inclusive
    parser.add_argument("--episodes_per_seed", type=int, default=1)
    parser.add_argument("--max_steps_per_episode", type=int, default=1600)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # DQN root is the folder containing this script
    DQN_ROOT = os.path.dirname(os.path.abspath(__file__))

    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_end + 1))

    for cfg in args.configs:
        # run all train seeds for this config
        for ts in args.train_seeds:
            print(f"\n=== Running eval: config={cfg} train_seed={ts} ===")
            try:
                run_eval(
                    config=cfg,
                    train_seed=ts,
                    num_bins=args.num_bins,
                    eval_seeds=eval_seeds,
                    episodes_per_seed=args.episodes_per_seed,
                    max_steps_per_episode=args.max_steps_per_episode,
                    device=args.device,
                )
            except FileNotFoundError as e:
                print(f"[SKIP] {cfg} seed{ts}: {e}")
                continue

        # after finishing all seeds for a config, aggregate whatever exists
        print(f"\n=== Aggregating: config={cfg} ===")
        summarize_config(DQN_ROOT, cfg)


if __name__ == "__main__":
    main()
