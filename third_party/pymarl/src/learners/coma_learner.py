import copy
import os
import csv
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop


class COMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # CSV logging of learner stats
        self.learner_csv_path = None
        if hasattr(args, "exp_dir") and args.exp_dir is not None:
            csv_dir = os.path.join(args.exp_dir, "csv_output")
            os.makedirs(csv_dir, exist_ok=True)
            self.learner_csv_path = os.path.join(csv_dir, "learner_stats.csv")

            if not os.path.exists(self.learner_csv_path):
                with open(self.learner_csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "t_env",
                        "episode_num",
                        "coma_loss",
                        "agent_grad_norm",
                        "critic_loss",
                        "critic_grad_norm",
                        "td_error_abs",
                        "q_taken_mean",
                        "target_mean",
                        "advantage_mean",
                        "pi_max",
                        "mask_sum"
                    ])


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t)

        actions = actions[:,:-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        denom = mac_out.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        mac_out = mac_out / denom
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        # log_pi_taken = th.log(pi_taken)
        log_pi_taken = th.log(pi_taken.clamp(min=1e-10))

        advantages = (q_taken - baseline).detach()

        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if self.learner_csv_path is not None:
            mask_elems = mask.sum().item()
            ts_logged = len(critic_train_stats["critic_loss"])

            if ts_logged > 0:
                critic_loss_mean = sum(critic_train_stats["critic_loss"]) / ts_logged
                critic_grad_norm_mean = sum(critic_train_stats["critic_grad_norm"]) / ts_logged
                td_error_abs_mean = sum(critic_train_stats["td_error_abs"]) / ts_logged
                q_taken_mean = sum(critic_train_stats["q_taken_mean"]) / ts_logged
                target_mean = sum(critic_train_stats["target_mean"]) / ts_logged
            else:
                critic_loss_mean = 0.0
                critic_grad_norm_mean = 0.0
                td_error_abs_mean = 0.0
                q_taken_mean = 0.0
                target_mean = 0.0

            advantage_mean = (advantages * mask).sum().item() / max(mask_elems, 1e-6)
            pi_max_mean = (pi.max(dim=1)[0] * mask).sum().item() / max(mask_elems, 1e-6)

            with open(self.learner_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    int(t_env),
                    int(episode_num),
                    float(coma_loss.item()),
                    float(grad_norm),
                    float(critic_loss_mean),
                    float(critic_grad_norm_mean),
                    float(td_error_abs_mean),
                    float(q_taken_mean),
                    float(target_mean),
                    float(advantage_mean),
                    float(pi_max_mean),
                    float(mask_elems),
                ])


        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])

            if ts_logged > 0:
                for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                    self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / max(mask_elems, 1e-6), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / max(mask_elems, 1e-6), t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        target_q_vals = self.target_critic(batch)[:, :]
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)

        q_vals = th.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic(batch, t)
            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            targets_t = targets[:, t]

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return q_vals, running_log

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path=None):
        if path is None:
            path = getattr(self.args, "model_dir", None) or "models"
        os.makedirs(path, exist_ok=True)

        self.mac.save_models(path)
        th.save(self.critic.state_dict(), os.path.join(path, "critic.th"))
        th.save(self.agent_optimiser.state_dict(), os.path.join(path, "agent_opt.th"))
        th.save(self.critic_optimiser.state_dict(), os.path.join(path, "critic_opt.th"))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(os.path.join(path, "critic.th"), map_location=lambda storage, loc: storage)
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(os.path.join(path, "agent_opt.th"), map_location=lambda storage, loc: storage)
        )
        self.critic_optimiser.load_state_dict(
            th.load(os.path.join(path, "critic_opt.th"), map_location=lambda storage, loc: storage)
        )


    def save_weights(self, path=None):
        """
        Weights-only checkpoint for eval:
        - mac weights
        - critic weights
        No optimizers.
        """
        if path is None:
            path = getattr(self.args, "model_dir", None) or "models"
        os.makedirs(path, exist_ok=True)

        self.mac.save_models(path)
        th.save(self.critic.state_dict(), os.path.join(path, "critic.th"))