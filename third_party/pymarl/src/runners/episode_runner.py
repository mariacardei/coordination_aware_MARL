from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from utils.csv_logger import CSVLogger
import os
import time
import sys
import shutil
import warnings
warnings.filterwarnings("once") 

def _fmt(x):
    # Compact float formatting
    if isinstance(x, float):
        if x == 0.0:
            return "0"
        if abs(x) < 1e-3:
            return f"{x:.0e}".replace("+", "")
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s
    return str(x)

def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-=+") else "_" for c in s)


def _env_arg(env_args, key, default=None):
    if isinstance(env_args, dict):
        return env_args.get(key, default)
    return getattr(env_args, key, default)


def build_qmix_exp_name(args) -> str:
    ea = args.env_args

    agents = _env_arg(ea, "agents")
    tasks = _env_arg(ea, "tasks")
    w = _env_arg(ea, "width")
    h = _env_arg(ea, "height")
    bins = _env_arg(ea, "num_bins")

    # ---- Required/common ----
    seed = getattr(args, "seed", 0)
    tmax = int(getattr(args, "t_max", 0))
    lr = getattr(args, "lr", None)
    gamma = getattr(args, "gamma", None)
    batch = getattr(args, "batch_size", None)  # learner batch_size
    tgt_upd = getattr(args, "target_update_interval", None)

    eps_final = getattr(args, "epsilon_finish", None)
    eps_anneal = getattr(args, "epsilon_anneal_time", None)
    eps_frac = getattr(args, "epsilon_frac", None)
    if eps_frac is None and eps_anneal is not None and tmax > 0:
        eps_frac = float(eps_anneal) / float(tmax)

    mixer = getattr(args, "mixer", None)  # qmix or vdn
    double_q = int(bool(getattr(args, "double_q", False)))

    # ---- QMIX-specific knobs (only if mixer=qmix) ----
    mix_emb = getattr(args, "mixing_embed_dim", None) if mixer == "qmix" else None
    h_layers = getattr(args, "hypernet_layers", None) if mixer == "qmix" else None
    h_emb = getattr(args, "hypernet_embed", None) if mixer == "qmix" else None

    # Optional: buffer_size
    # buf = getattr(args, "buffer_size", None)

    parts = []
    parts.append(f"seed{seed}")
    if None not in (agents, tasks, w, h):
        parts.append(f"{int(agents)}agents")
        parts.append(f"{int(tasks)}tasks")
        parts.append(f"{int(w)}x{int(h)}")
    if bins is not None:
        parts.append(f"{int(bins)}bins")
    parts.append(f"{tmax}steps")

    if lr is not None:
        parts.append(f"{_fmt(lr)}lr")
    if gamma is not None:
        parts.append(f"{_fmt(gamma)}gamma")
    if tgt_upd is not None:
        parts.append(f"{int(tgt_upd)}targetupdate")
    if batch is not None:
        parts.append(f"{int(batch)}batch")

    if eps_frac is not None:
        parts.append(f"{_fmt(eps_frac)}epsfrac")
    if eps_anneal is not None:
        parts.append(f"{int(eps_anneal)}epsanneal")
    if eps_final is not None:
        parts.append(f"{_fmt(eps_final)}epsfinal")

    if mixer is not None:
        parts.append(str(mixer))
    if mixer == "qmix":
        if mix_emb is not None:
            parts.append(f"{int(mix_emb)}mixemb")
        if h_layers is not None:
            parts.append(f"{int(h_layers)}hyper")
        if h_emb is not None:
            parts.append(f"{int(h_emb)}hyperemb")

    parts.append(f"doubleq{double_q}")

    return _safe("_".join(parts))

def build_qmix_base_dir(args) -> str:
    ea = args.env_args
    agents = _env_arg(ea, "agents")
    tasks = _env_arg(ea, "tasks")
    w = _env_arg(ea, "width")
    h = _env_arg(ea, "height")

    if None not in (agents, tasks, w, h):
        return _safe(f"{int(agents)}agents_{int(tasks)}tasks_{int(w)}x{int(h)}")
    # fallback
    return _safe(str(getattr(args, "name", "QMIX")))


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0
        self.run_start_time = time.time()

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.eval_returns = []
        self.eval_steps = []
        self.eval_coord_stats = []
        self.eval_ep_in_block = 0  # 0..test_nepisode-1, resets after summary


        # Build a deterministic experiment directory 
        base_dir = os.path.join(self.args.local_results_path, build_qmix_base_dir(self.args))
        run_dir_name = build_qmix_exp_name(self.args)
        self.exp_dir = os.path.join(base_dir, run_dir_name)

        def _reset_dir(path: str):
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        # reset everything so reruns don't mix outputs
        _reset_dir(self.exp_dir)
        _reset_dir(os.path.join(self.exp_dir, "logs"))
        _reset_dir(os.path.join(self.exp_dir, "tb_logs"))
        _reset_dir(os.path.join(self.exp_dir, "saved_models"))
        _reset_dir(os.path.join(self.exp_dir, "csv_output"))


        # expose to args for learner/logger saving
        self.args.exp_dir = self.exp_dir
        self.args.model_dir = os.path.join(self.exp_dir, "saved_models")
        self.args.tb_dir = os.path.join(self.exp_dir, "tb_logs")
        self.args.logs_dir = os.path.join(self.exp_dir, "logs")

        csv_dir = os.path.join(self.exp_dir, "csv_output")

        TRAIN_HEADER = [
            "TotalSteps","Episode","Reward","StepsInEpisode","AvgLoss",
            "Epsilon","Time_sec","WallTimeTotal_sec",
            "J_upper_median","ForcedIdle_sum","UniqueTasksAssigned_mean",
            "DeterministicAgents_mean","NumConflicts_sum","StepsPerSec"
        ]

        EVAL_EP_HEADER = [
            "TotalSteps","TrainEpisode","EvalEpisodeIdx","EvalSeed",
            "Reward","StepsInEpisode","Epsilon","Time_sec",
            "J_upper_median","ForcedIdle_sum","UniqueTasksAssigned_mean",
            "DeterministicAgents_mean","NumConflicts_sum","StepsPerSec"
        ]

        EVAL_SUM_HEADER = [
            "TotalSteps","TrainEpisode","NEvalEpisodes",
            "Reward_mean","Reward_std",
            "Steps_mean","Steps_std",
            "J_upper_median_mean","J_upper_median_std",
            "ForcedIdle_sum_mean","ForcedIdle_sum_std",
            "UniqueTasksAssigned_mean","UniqueTasksAssigned_std",
            "DeterministicAgents_mean","DeterministicAgents_std",
            "NumConflicts_sum_mean","NumConflicts_sum_std",
        ]


        self.csv_logger = CSVLogger(csv_dir, filename="train_results.csv", header=TRAIN_HEADER)
        self.eval_csv_logger = CSVLogger(csv_dir, filename="eval_results.csv", header=EVAL_EP_HEADER)
        self.eval_summary_logger = CSVLogger(csv_dir, filename="eval_summary.csv", header=EVAL_SUM_HEADER)

        log_path = os.path.join(self.exp_dir, "logs", "stdout_stderr.log")

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        class Tee:
            def __init__(self, *streams):
                self.streams = list(streams)

            def write(self, data):
                for s in self.streams:
                    if s is None:
                        continue
                    try:
                        s.write(data)
                        s.flush()
                    except Exception:
                        pass

            def flush(self):
                for s in self.streams:
                    if s is None:
                        continue
                    try:
                        s.flush()
                    except Exception:
                        pass

            def close(self):
                # logging.shutdown() expects this to exist
                for s in self.streams:
                    if s is None:
                        continue
                    try:
                        s.flush()
                    except Exception:
                        pass
                    # Only close real file handles you opened (not sys.__stdout__/__stderr__)
                    try:
                        if s not in (sys.__stdout__, sys.__stderr__):
                            s.close()
                    except Exception:
                        pass



        self._log_f = open(log_path, "w", buffering=1)
        self._log_f.write(f"\n===== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        sys.stdout = Tee(sys.__stdout__, self._log_f)
        sys.stderr = Tee(sys.__stderr__, self._log_f)
        print(f"[Logging] Writing stdout/stderr to: {log_path}", flush=True)

        try:
            import torch
            print("CUDA available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass


        self.episode = 0
        self.episode_start_time = None

        # Log the first run
        self.log_train_stats_t = -1000000


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()
        # Restore stdout/stderr FIRST
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass

        # Now close the log file
        try:
            if hasattr(self, "_log_f") and self._log_f is not None:
                self._log_f.flush()
                self._log_f.close()
        except Exception:
            pass

    def _get_eval_seed(self, eval_idx: int) -> int:
        """
        Deterministic eval seeds per (train_seed, eval_episode_idx).
        If train seed = s, eval seeds = s*1000 + {0..test_nepisode-1}
        """
        base = int(getattr(self.args, "seed", 0)) * 1000
        return base + int(eval_idx)

    def reset(self, seed=None):
        self.batch = self.new_batch()
        # try to seed env per episode if supported
        try:
            if seed is None:
                self.env.reset()
            else:
                self.env.reset(seed=seed)
        except TypeError:
            # fallback: env.reset() doesn't accept seed
            # if env has set_seed:
            if seed is not None and hasattr(self.env, "set_seed"):
                self.env.set_seed(seed)
            self.env.reset()

        self.t = 0
        self.episode_start_time = time.time()


    def run(self, test_mode=False):
        eval_seed = None
        eval_episode_idx = None

        if test_mode:
            eval_episode_idx = int(self.eval_ep_in_block)
            eval_seed = self._get_eval_seed(eval_episode_idx)

        self.reset(seed=eval_seed)

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # reward, terminated, env_info = self.env.step(actions[0])
            # Normalize to python ints of shape (n_agents,)
            a0 = actions[0]
            try:
                # torch tensor -> numpy
                if hasattr(a0, "detach"):
                    a0 = a0.detach().cpu().numpy()
                # squeeze last dim if it's (n_agents, 1)
                a0 = np.asarray(a0).reshape(-1)
                a0 = [int(x) for x in a0.tolist()]
            except Exception:
                # fallback
                a0 = [int(x) for x in list(a0)]

            reward, terminated, env_info = self.env.step(a0)

            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # ---- Episode-level metrics (train + eval) ----
        env_stats = self.env.get_stats()

        episode_time = time.time() - self.episode_start_time
        steps_in_episode = self.t
        steps_per_sec = steps_in_episode / max(episode_time, 1e-6)


        if not test_mode:
            self.t_env += self.t
        cur_returns.append(episode_return)
        if test_mode:

            total_steps = int(self.t_env)          # training steps at which eval happens
            train_ep_idx = int(self.episode)       # training episode counter
            eps = 0.0

            env_stats = self.env.get_stats()

            self.eval_returns.append(float(episode_return))
            self.eval_steps.append(int(steps_in_episode))
            self.eval_coord_stats.append({
                "J_upper_median": float(env_stats.get("J_upper_median", 0.0)),
                "ForcedIdle_sum": int(env_stats.get("ForcedIdle_sum", 0)),
                "UniqueTasksAssigned_mean": float(env_stats.get("UniqueTasksAssigned_mean", 0.0)),
                "DeterministicAgents_mean": float(env_stats.get("DeterministicAgents_mean", 0.0)),
                "NumConflicts_sum": int(env_stats.get("NumConflicts_sum", 0)),
            })

            # per-episode eval row
            self.eval_csv_logger.log([
                total_steps,
                train_ep_idx,
                int(eval_episode_idx),
                int(eval_seed),
                float(episode_return),
                int(steps_in_episode),
                float(eps),
                float(episode_time),
                float(env_stats.get("J_upper_median", 0.0)),
                int(env_stats.get("ForcedIdle_sum", 0)),
                float(env_stats.get("UniqueTasksAssigned_mean", 0.0)),
                float(env_stats.get("DeterministicAgents_mean", 0.0)),
                int(env_stats.get("NumConflicts_sum", 0)),
                float(steps_per_sec),
            ])
            self.eval_ep_in_block += 1



        if test_mode and (len(self.eval_returns) == self.args.test_nepisode):
            total_steps = int(self.t_env)
            train_ep_idx = int(self.episode)

            rets = np.asarray(self.eval_returns, dtype=float)
            steps = np.asarray(self.eval_steps, dtype=float)

            coord = self.eval_coord_stats
            def _mean(key): return float(np.mean([d[key] for d in coord]))
            def _std(key): 
                return float(np.std([d[key] for d in coord], ddof=1))

            # summary row
            self.eval_summary_logger.log([
                total_steps,
                train_ep_idx,
                int(self.args.test_nepisode),
                float(np.mean(rets)), float(np.std(rets)),
                float(np.mean(steps)), float(np.std(steps)),
                _mean("J_upper_median"), _std("J_upper_median"),
                _mean("ForcedIdle_sum"), _std("ForcedIdle_sum"),
                _mean("UniqueTasksAssigned_mean"), _std("UniqueTasksAssigned_mean"),
                _mean("DeterministicAgents_mean"), _std("DeterministicAgents_mean"),
                _mean("NumConflicts_sum"), _std("NumConflicts_sum"),
            ])


           # TensorBoard (one point per eval point)
            self.logger.log_stat("eval/Reward_mean", float(np.mean(rets)), total_steps, to_sacred=False)
            self.logger.log_stat("eval/Reward_std",  float(np.std(rets,ddof=1)),  total_steps, to_sacred=False)
            self.logger.log_stat("eval/Steps_mean",  float(np.mean(steps)), total_steps, to_sacred=False)
            self.logger.log_stat("eval/Steps_std",   float(np.std(steps,ddof=1)),  total_steps, to_sacred=False)

            self.logger.log_stat("eval/J_upper_median_mean", _mean("J_upper_median"), total_steps, to_sacred=False)
            self.logger.log_stat("eval/ForcedIdle_sum_mean", _mean("ForcedIdle_sum"), total_steps, to_sacred=False)
            self.logger.log_stat("eval/UniqueTasksAssigned_mean", _mean("UniqueTasksAssigned_mean"), total_steps, to_sacred=False)
            self.logger.log_stat("eval/DeterministicAgents_mean", _mean("DeterministicAgents_mean"), total_steps, to_sacred=False)
            self.logger.log_stat("eval/NumConflicts_sum_mean", _mean("NumConflicts_sum"), total_steps, to_sacred=False)

            # reset for next eval point
            self.eval_returns.clear()
            self.eval_steps.clear()
            self.eval_coord_stats.clear()
            self.eval_ep_in_block = 0


        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # ---- CSV + TensorBoard logging (training only) ----
        if not test_mode:
            env_stats = self.env.get_stats()

            episode_time = time.time() - self.episode_start_time
            steps_in_episode = self.t
            steps_per_sec = steps_in_episode / max(episode_time, 1e-6)
            wall_time_total_sec = time.time() - self.run_start_time

            epsilon = (
                float(self.mac.action_selector.epsilon)
                if hasattr(self.mac.action_selector, "epsilon")
                else float(getattr(self.args, "epsilon_finish", 0.0))
            )

            total_steps = int(self.t_env)
            ep_idx = int(self.episode)

            # ---- CSV row ----
            row = [
                total_steps,                           # TotalSteps
                ep_idx,                                # Episode
                float(episode_return),                 # Reward
                int(steps_in_episode),                 # StepsInEpisode
                0.0,                                   # AvgLoss (episode-level; QMIX loss is batch-level)
                float(epsilon),                        # Epsilon
                float(episode_time),                   # Time_sec
                float(wall_time_total_sec),            # WallTimeTotal_sec (cumulative)
                float(env_stats.get("J_upper_median", 0.0)),
                int(env_stats.get("ForcedIdle_sum", 0)),
                float(env_stats.get("UniqueTasksAssigned_mean", 0.0)),
                float(env_stats.get("DeterministicAgents_mean", 0.0)),
                int(env_stats.get("NumConflicts_sum", 0)),
                float(steps_per_sec),
            ]
            self.csv_logger.log(row)

            # ---- TensorBoard scalars ----
            # Prefix tags so TB is organized and matches CSV columns
            self.logger.log_stat("train/Reward", float(episode_return), total_steps, to_sacred=False)
            self.logger.log_stat("train/StepsInEpisode", int(steps_in_episode), total_steps, to_sacred=False)
            self.logger.log_stat("train/Epsilon", float(epsilon), total_steps, to_sacred=False)
            self.logger.log_stat("train/Time_sec", float(episode_time), total_steps, to_sacred=False)
            self.logger.log_stat("train/StepsPerSec", float(steps_per_sec), total_steps, to_sacred=False)
            self.logger.log_stat("train/WallTimeTotal_sec", float(wall_time_total_sec), total_steps, to_sacred=False)

            self.logger.log_stat("coord/J_upper_median", float(env_stats.get("J_upper_median", 0.0)), total_steps, to_sacred=False)
            self.logger.log_stat("coord/ForcedIdle_sum", int(env_stats.get("ForcedIdle_sum", 0)), total_steps, to_sacred=False)
            self.logger.log_stat("coord/UniqueTasksAssigned_mean", float(env_stats.get("UniqueTasksAssigned_mean", 0.0)), total_steps, to_sacred=False)
            self.logger.log_stat("coord/DeterministicAgents_mean", float(env_stats.get("DeterministicAgents_mean", 0.0)), total_steps, to_sacred=False)
            self.logger.log_stat("coord/NumConflicts_sum", int(env_stats.get("NumConflicts_sum", 0)), total_steps, to_sacred=False)

            # Keep per-episode console line
            exp_name = getattr(self.args, "name", "QMIX")

            print(
                f"Exp: {exp_name}, Episode {ep_idx + 1} Done -- "
                f"Reward: {episode_return}, Steps: {steps_in_episode}, Avg loss = (not calculated), "
                f"TotalSteps={total_steps}/{int(getattr(self.args, 't_max', total_steps))}, "
                f"WallTimeTotal_sec={wall_time_total_sec:.2f}",
                flush=True,
            )


            self.episode += 1


        return self.batch

    def _log(self, returns, stats, prefix):
        returns.clear()
        stats.clear()
