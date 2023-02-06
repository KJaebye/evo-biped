# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class EvoBipedalWalkerAgent
#   @author: by Kangyao Huang
#   @created date: 02.Feb.2023
# ------------------------------------------------------------------------------------------------------------------- #


import os
import gym
import pickle
from lib.core.zfilter import ZFilter
from lib.utils.torch import *
from lib.agents.agent_ppo2 import AgentPPO2

from custom.models.evo_bipedalwalker_policy import EvoBipedalWalkerPolicy
from custom.models.evo_bipedalwalker_critic import EvoBipedalWalkerValue


class Filter(ZFilter):
    def __init__(self, scale_state_dim, control_state_dim, demean=True, destd=True,
                 scale_clip=0.75, control_clip=10.0):
        self.fixed_dim = scale_state_dim
        self.control_clip = control_clip
        self.scale_clip = scale_clip
        super(Filter, self).__init__(control_state_dim, demean=demean, destd=destd, clip=control_clip)

    def __call__(self, x, update=True):
        x_fixed = x[:self.fixed_dim]
        x = x[self.fixed_dim:]
        if update and not self.fix:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.control_clip, self.control_clip)
        x_fixed = np.clip(x_fixed, 1 - self.scale_clip, 1 + self.scale_clip)
        x = np.concatenate((x_fixed, x))
        return x

class EvoBipedalWalkerAgent(AgentPPO2):
    def __init__(self, task, domain, cfg, logger, dtype, device, num_threads, training=True, checkpoint=0, mean_action=False):
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint
        self.setup_env()

        super().__init__(self.cfg, self.env, self.logger, self.dtype, self.device, self.num_threads,
                         training=self.training, checkpoint=checkpoint, mean_action=mean_action)

    def setup_env(self):
        from custom.envs.evo_bipedalwalker import EvoBipedalWalker
        self.env = EvoBipedalWalker()
        self.env.action_space.seed(42)

        if not self.training:
            self.wrap_env_monitor()

    def wrap_env_monitor(self):
        if not self.training:
            output_dir = './tmp/%s/%s/%s/' % (self.cfg.domain, self.cfg.task, self.cfg.rec)
            path = os.path.join(output_dir, 'video')
            path = os.path.abspath(path)
            if not os.path.exists(path):
                os.makedirs(path)
            self.env = gym.wrappers.RecordVideo(self.env, path)

    def setup_networks(self):
        self.running_state = Filter(self.env.scale_state_dim, self.env.control_state_dim, scale_clip=1, control_clip=5)

        """define actor and critic"""
        self.policy_net = EvoBipedalWalkerPolicy(self.cfg, self)
        self.value_net = EvoBipedalWalkerValue(self.cfg, self)

        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)

    def save_checkpoint(self, iter, log, log_eval):
        def save(checkpoint_path):
            to_device(torch.device('cpu'), self.policy_net, self.value_net)
            model_checkpoint = \
                {
                    'policy_dict': self.policy_net.state_dict(),
                    'value_dict': self.value_net.state_dict(),
                    'running_state': self.running_state,
                    'best_reward': self.best_reward,
                    'iter': iter
                }
            pickle.dump(model_checkpoint, open(checkpoint_path, 'wb'))
            to_device(self.device, self.policy_net, self.value_net)

        cfg = self.cfg

        if cfg.save_model_interval > 0 and (iter + 1) % cfg.save_model_interval == 0:
            self.tb_logger.flush()
            self.logger.critical(f'Saving the interval checkpoint with rewards {self.best_reward:.2f}')
            save('%s/iter_%04d.p' % (cfg.model_dir, iter + 1))

        if log_eval['avg_reward'] > self.best_reward:
            self.best_reward = log_eval['avg_reward']
            self.save_best_flag = True
            self.logger.critical('Get the best episode reward: {:.2f}'.format(self.best_reward))

        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.critical(f'Saving the best checkpoint with rewards {self.best_reward:.2f}')
            save('%s/best.p' % self.cfg.model_dir)
            self.save_best_flag = False
