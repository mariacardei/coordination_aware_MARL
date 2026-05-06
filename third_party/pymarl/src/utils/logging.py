from collections import defaultdict
import logging
import numpy as np
import torch as th


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        from torch.utils.tensorboard import SummaryWriter
        self._tb_writer = SummaryWriter(log_dir=directory_name)
        self.use_tb = True


    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            v = value.detach().cpu().item() if th.is_tensor(value) else float(value)
            self._tb_writer.add_scalar(key, v, t)



        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            # item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            vals = [x[1] for x in self.stats[k][-window:]]

            # Convert torch tensors (cpu or cuda) to Python floats
            vals = [
                v.detach().cpu().item() if th.is_tensor(v)
                else float(v)
                for v in vals
            ]

            item = "{:.4f}".format(np.mean(vals))

            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

    def close(self):
        if getattr(self, "_tb_writer", None) is not None:
            try:
                self._tb_writer.flush()
            except Exception:
                pass
            self._tb_writer.close()
            self._tb_writer = None


def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

