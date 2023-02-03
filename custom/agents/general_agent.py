# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class GeneralAgent
#   @author: by Kangyao Huang
#   @created date: 24.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    This agent is for all dm_control suite domains and tasks.
"""

import numpy as np
import torch
import os
import gym
from lib.agents.agent_ppo2 import AgentPPO2


class GeneralAgent(AgentPPO2):
    def __init__(self, task, domain, cfg, logger, dtype, device, num_threads, training=True, checkpoint=0):
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint
        self.setup_env()

        super().__init__(self.cfg, self.env, self.logger, self.dtype, self.device, self.num_threads,
                         training=self.training, checkpoint=checkpoint)

    def setup_env(self):
        if self.training:
            self.env = gym.make('BipedalWalker-v3')
        else:
            self.env = gym.make('BipedalWalker-v3', render_mode="human")
        # self.env.action_space.seed(42)
        self.wrap_env_monitor()

    def wrap_env_monitor(self):
        if not self.training:
            output_dir = './tmp/%s/%s/%s/' % (self.cfg.domain, self.cfg.task, self.cfg.rec)
            path = os.path.join(output_dir, 'video')
            path = os.path.abspath(path)
            if not os.path.exists(path):
                os.makedirs(path)
            self.env = gym.wrappers.RecordVideo(self.env, path)