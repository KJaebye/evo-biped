"""
    This uses only one policy to learn scale vector and control.
"""

import torch.nn as nn
from lib.core.running_norm import RunningNorm
from lib.models.mlp import MLP

class EvoBipedalWalkerValue(nn.Module):
    def __init__(self, cfg, agent):
        super(EvoBipedalWalkerValue, self).__init__()
        self.cfg = cfg
        self.agent = agent
        # dimension define
        self.scale_vector = agent.env.scale_vector
        self.control_action_dim = agent.env.control_action_dim
        self.control_state_dim = agent.env.control_state_dim
        self.entire_action_dim = agent.env.entire_action_dim
        self.entire_state_dim = agent.env.entire_state_dim

        self.norm = RunningNorm(self.entire_state_dim)
        cur_dim = self.entire_state_dim
        self.mlp = MLP(cur_dim,
                       hidden_dims=self.cfg.value_spec['mlp'],
                       activation=self.cfg.value_spec['htype'])
        cur_dim = self.mlp.out_dim
        self.value_head = nn.Linear(cur_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)
        value = self.value_head(x)
        return value
