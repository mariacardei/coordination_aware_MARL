import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from datetime import datetime

import sys
import signal
import atexit
import shutil



def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # # Graceful exit so TB/log files flush to disk
    # import sys
    # sys.exit(0)
    return



def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    # ---------------------------------------
    # TensorBoard: setup AFTER runner resets dirs
    # ---------------------------------------
    if getattr(args, "use_tensorboard", False):
        # runner sets args.tb_dir = <exp_dir>/tb_logs
        logger.setup_tb(args.tb_dir)
        # Force creation of an events file immediately
        logger.log_stat("tb/init", 0, 0, to_sacred=False)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()


    # -----------------------------
    # Save-on-death
    # -----------------------------
    interrupt_state = {"handled": False}

    def _ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

    def _safe_save_checkpoint(tag: str, signum: int | None = None):
        """
        Saves BOTH:
          1) resume-capable model (includes optimizer) to saved_models/<tag>_step<t_env>/
          2) weights-only to saved_models/latest_weights/
        And also writes/updates:
          - saved_models/latest_payload/ (copy of the resume-capable dir)
        """
        try:
            base_model_dir = getattr(args, "model_dir", None)
            if base_model_dir is None:
                # fallback (but in your code runner sets args.model_dir)
                base_model_dir = os.path.join(args.local_results_path, args.name, f"seed{args.seed}", "saved_models")
            _ensure_dir(base_model_dir)

            t_env = int(getattr(runner, "t_env", 0))
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sig_part = f"_sig{signum}" if signum is not None else ""
            step_dirname = f"{tag}{sig_part}_step{t_env}_{stamp}"

            # 1) Regular / resume-capable save
            payload_dir = os.path.join(base_model_dir, step_dirname)
            _ensure_dir(payload_dir)
            learner.save_models(payload_dir)

            # Maintain a "latest_payload" folder (overwrite-safe)
            latest_payload_dir = os.path.join(base_model_dir, "latest_payload")
            if os.path.isdir(latest_payload_dir):
                shutil.rmtree(latest_payload_dir)
            shutil.copytree(payload_dir, latest_payload_dir)

            # 2) Weights-only save (overwrite-safe)
            latest_weights_dir = os.path.join(base_model_dir, "latest_weights")
            if os.path.isdir(latest_weights_dir):
                shutil.rmtree(latest_weights_dir)
            _ensure_dir(latest_weights_dir)

            # requires you added learner.save_weights()
            if hasattr(learner, "save_weights"):
                learner.save_weights(latest_weights_dir)
            else:
                # fallback: at least save regular again
                learner.save_models(latest_weights_dir)

            logger.console_logger.info(
                f"[Checkpoint] Saved payload -> {payload_dir} | "
                f"latest_payload -> {latest_payload_dir} | latest_weights -> {latest_weights_dir}"
            )

            # Flush TB writer if present
            try:
                logger.close()
            except Exception:
                pass

            # Flush stdout/stderr (like FDQN)
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass

        except Exception as e:
            try:
                logger.console_logger.error(f"[Checkpoint] FAILED ({tag}): {e}")
            except Exception:
                print(f"[Checkpoint] FAILED ({tag}): {e}", flush=True)

    def _handle_interrupt(signum, frame):
        if interrupt_state["handled"]:
            return
        interrupt_state["handled"] = True
        try:
            logger.console_logger.info(f"\n[Signal] Caught signal {signum}. Saving checkpoint before exit...")
        except Exception:
            print(f"\n[Signal] Caught signal {signum}. Saving checkpoint before exit...", flush=True)

        _safe_save_checkpoint(tag="interrupted", signum=signum)

        # mirror FDQN: exit code 128 + signum
        raise SystemExit(128 + signum)

    # Catch SLURM termination & Ctrl-C
    signal.signal(signal.SIGTERM, _handle_interrupt)
    signal.signal(signal.SIGINT, _handle_interrupt)

    def _atexit_save():
        if interrupt_state["handled"]:
            return
        _safe_save_checkpoint(tag="atexit")

    atexit.register(_atexit_save)




    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env

            # Save inside: qmix_training_experiments/QMIX.../seed0/saved_models/<t_env>/
            base_model_dir = getattr(args, "model_dir", None)
            if base_model_dir is None:
                base_model_dir = os.path.join(args.local_results_path, "models")  # fallback

            save_path = os.path.join(base_model_dir, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)

            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)


        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # # Always save final model
    # final_path = os.path.join(
    #     args.local_results_path,
    #     args.name,
    #     f"seed{args.seed}",
    #     "saved_models",
    #     "final"
    # )
    # os.makedirs(final_path, exist_ok=True)
    # logger.console_logger.info(f"Saving FINAL model to {final_path}")
    # learner.save_models(final_path)


    # Always save final model INSIDE the run directory (FDQN-style)
    final_path = os.path.join(args.model_dir, "final")  # args.model_dir = <exp_dir>/saved_models
    os.makedirs(final_path, exist_ok=True)
    logger.console_logger.info(f"Saving FINAL model to {final_path}")
    learner.save_models(final_path)



    runner.close_env()
    logger.console_logger.info("Finished Training")

    # Close TB writer cleanly
    try:
        logger.close()
    except Exception:
        pass



def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    # If epsilon_frac is provided, override anneal time to match step budget
    if "epsilon_frac" in config and config["epsilon_frac"] is not None:
        config["epsilon_anneal_time"] = max(1, int(config["epsilon_frac"] * config["t_max"]))


    return config
