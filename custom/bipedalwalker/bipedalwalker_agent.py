# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class BipedalWalkerAgent
#   @author: by Kangyao Huang
#   @created date: 02.Feb.2023
# ------------------------------------------------------------------------------------------------------------------- #

import os
import gym
from lib.agents.agent_ppo2 import AgentPPO2


class BipedalWalkerAgent(AgentPPO2):
    def __init__(self, task, domain, cfg, logger, dtype, device, num_threads, training=True, checkpoint=0, mean_action=False):
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint
        self.setup_env(task)

        if not self.training:
            self.wrap_env_monitor()

        super().__init__(self.cfg, self.env, self.logger, self.dtype, self.device, self.num_threads,
                         training=self.training, checkpoint=checkpoint, mean_action=mean_action)

    def setup_env(self, task):
        from custom.bipedalwalker.bipedalwalker import AugmentBipedalWalker
        from custom.bipedalwalker.bipedalwalker import AugmentBipedalWalkerHardcore
        if task == 'easy':
            self.env = AugmentBipedalWalker(self.logger)
        else:
            self.env = AugmentBipedalWalkerHardcore(self.logger)
        self.env.action_space.seed(self.cfg.seed)

    def wrap_env_monitor(self):
        if not self.training:
            output_dir = './tmp/%s/%s/%s/' % (self.cfg.domain, self.cfg.task, self.cfg.rec)
            path = os.path.join(output_dir, 'video')
            path = os.path.abspath(path)
            if not os.path.exists(path):
                os.makedirs(path)
            self.env = gym.wrappers.RecordVideo(self.env, path)
