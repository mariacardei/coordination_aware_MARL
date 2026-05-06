# PyMARL Integration

This repository vendors PyMARL under `third_party/pymarl/` so anyone can run IQL, VDN, QMIX, QTRAN, and COMA without chasing a separate framework checkout.

STAT is the environment. There is one core model and two API wrappers:

- Core task-allocation dynamics live in `src/stat_env/stat_model.py`.
- Gymnasium-facing construction lives in `src/stat_env/stat_gym.py` and `src/stat_env/gymnasium_env.py`. Use this for standalone methods, DQN, FDQN, and new algorithms that want a normal Python environment object.
- PyMARL-facing construction lives in `src/stat_env/pymarl_env.py`. Use this only when running methods through PyMARL, because PyMARL expects methods like `get_obs()`, `get_state()`, `get_avail_actions()`, and `get_env_info()`.
- The vendored PyMARL registry shim at `third_party/pymarl/src/envs/stat_env.py` imports the STAT PyMARL wrapper and exposes it under PyMARL's expected environment name.

The current integration intentionally keeps PyMARL as third-party runner code rather than rewriting it into the STAT package. That makes the release easier to audit: environment behavior is owned by `stat_env`, while PyMARL owns only its algorithms, learners, runners, and experiment configuration.

PyMARL smoke configs use `agents` and `tasks`, matching the public STAT API.
