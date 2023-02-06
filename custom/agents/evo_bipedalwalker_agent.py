# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class EvoBipedalWalkerAgent
#   @author: by Kangyao Huang
#   @created date: 02.Feb.2023
# ------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import torch
import os
import gym
import multiprocessing
import time
import math
from lib.core.zfilter import ZFilter
from lib.utils.torch import *
from lib.agents.agent_ppo2 import AgentPPO2
from lib.core.memory import Memory
from custom.models.evo_bipedalwalker_policy import EvoBipedalWalkerPolicy
from custom.models.evo_bipedalwalker_critic import EvoBipedalWalkerValue


class EvoBipedalWalkerAgent(AgentPPO2):
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
        self.running_state = ZFilter((self.env.entire_state_dim), clip=5)

        """define actor and critic"""
        self.policy_net = EvoBipedalWalkerPolicy(self.cfg, self)
        self.value_net = EvoBipedalWalkerValue(self.cfg, self)

        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)
