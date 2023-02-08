
import torch
import math
import numpy as np
import torch.nn as nn
from lib.core.running_norm import RunningNorm
from lib.models.mlp import MLP

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


class EvoBipedalWalkerPolicy(nn.Module):
    def __init__(self, cfg, agent):
        super(EvoBipedalWalkerPolicy, self).__init__()
        self.cfg = cfg
        self.agent = agent
        # dimension define
        self.scale_state_dim = agent.env.scale_vector.size
        self.control_action_dim = agent.env.control_action_dim
        self.control_state_dim = agent.env.control_state_dim
        self.entire_action_dim = agent.env.entire_action_dim
        self.entire_state_dim = agent.env.entire_state_dim

        # scale transform
        self.scale_norm = RunningNorm(self.scale_state_dim)
        cur_dim = self.scale_state_dim
        self.scale_mlp = MLP(cur_dim,
                             hidden_dims=self.cfg.policy_spec['scale_mlp'],
                             activation=self.cfg.policy_spec['scale_htype'])
        cur_dim = self.scale_mlp.out_dim
        self.scale_state_mean = nn.Linear(cur_dim, self.scale_state_dim)
        self.scale_state_mean.weight.data.mul_(0.1)
        self.scale_state_mean.bias.data.mul_(0.0)
        self.scale_state_log_std = nn.Parameter(
            torch.ones(1, self.scale_state_dim) * self.cfg.policy_spec['scale_log_std'])

        # execution
        self.control_norm = RunningNorm(self.entire_state_dim)
        cur_dim = self.entire_state_dim
        self.control_mlp = MLP(cur_dim,
                               hidden_dims=self.cfg.policy_spec['control_mlp'],
                               activation=self.cfg.policy_spec['control_htype'])
        cur_dim = self.control_mlp.out_dim
        self.control_action_mean = nn.Linear(cur_dim, self.control_action_dim)
        # self.control_action_mean.weight.data.mul_(1)
        # self.control_action_mean.bias.data.mul_(0.0)
        self.control_action_log_std = nn.Parameter(
            torch.ones(1, self.control_action_dim) * self.cfg.policy_spec['control_log_std'])


        self.is_disc_action = False

    def forward(self, x):
        """
        This forward function is for evaluation process which uses the mean action option.
        :param x: 32 dimension tensor.
        :return: 12 dimension tensor including scale vector and execution action.
        """
        if self.agent.env.stage == 'scale_transform':
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            Important:   Using network to generate a morphology while evaluating.
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            # random scale vector
            random_scale_vector = self.agent.env.action_space.sample()[:self.scale_state_dim]
            x = self.scale_norm(torch.tensor(random_scale_vector))
            x = self.scale_mlp(x)
            scale_state_mean = self.scale_state_mean(x)
            # limit the scale vector to [, ]
            scale_state_mean = torch.ones([1, 8], dtype=torch.float) + scale_state_mean * 0.75
            scale_state_log_std = self.scale_state_log_std.expand_as(scale_state_mean).exp()
            scale_state_std = torch.exp(scale_state_log_std)

            control_action = np.array([0, 0, 0, 0])
            action = np.concatenate((scale_state_mean.squeeze(0), control_action))
            action = torch.tensor(action).unsqueeze(0).unsqueeze(0)
            return action

        if self.agent.env.stage == 'execution':
            scale_vector = x.squeeze(0)[:self.scale_state_dim].numpy()
            assert scale_vector.all() == self.agent.env.scale_vector.all()

            x = self.control_norm(x)
            x = self.control_mlp(x)
            control_action_mean = self.control_action_mean(x)
            control_action_log_std = self.control_action_log_std.expand_as(control_action_mean).exp()
            control_action_std = torch.exp(control_action_log_std)
            action_mean = torch.tensor(np.concatenate((scale_vector, control_action_mean[0]))).unsqueeze(0)

            return action_mean, control_action_log_std, control_action_std


    def _forward(self, x):
        """
        This forward function is for training process. Only used for execution stage since scale transform does not need
        forward propagation.
        :param x: 32 dimension tensor.
        :return: 4 dimension tensor for 'execution stage'.
        """
        if self.agent.env.stage == 'scale_transform':
            x = x[:self.scale_state_dim]
            x = self.scale_norm(torch.tensor(x))
            x = self.scale_mlp(x)
            scale_state_mean = self.scale_state_mean(x)
            # limit the scale vector to [, ]
            scale_state_mean = torch.ones([1, 8], dtype=torch.float) + scale_state_mean * 0.75
            scale_state_log_std = self.scale_state_log_std.expand_as(scale_state_mean).exp()
            scale_state_std = torch.exp(scale_state_log_std)
            return scale_state_mean, scale_state_log_std, scale_state_std

        if self.agent.env.stage == 'execution':
            x = self.control_norm(x)
            x = self.control_mlp(x)
            control_action_mean = self.control_action_mean(x)
            control_action_log_std = self.control_action_log_std.expand_as(control_action_mean).exp()
            control_action_std = torch.exp(control_action_log_std)
            return control_action_mean, control_action_log_std, control_action_std


    def select_action(self, x):
        """
        :param x: the input is the state of RL
        :return: return the action of RL. The scale vector is listed at first then control action.
        """
        if self.agent.env.stage == 'scale_transform':

            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            Important:    Select a random morphology while training. (mean_action==False)
                          Using network to generate a morphology while evaluating. (mean_action==True)
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            random_scale_vector = self.agent.env.action_space.sample()[:self.scale_state_dim]
            # scale_vector = np.clip(np.random.normal(1, 0.5, size=8), 0.25, 1.75)
            scale_vector = random_scale_vector

            control_action = np.array([0, 0, 0, 0])
            action = np.concatenate((scale_vector, control_action))
            action = torch.tensor(action).unsqueeze(0)
            return action

        elif self.agent.env.stage == 'execution':
            scale_vector = x.squeeze(0)[:self.scale_state_dim].numpy()
            assert scale_vector.all() == self.agent.env.scale_vector.all()

            control_action_mean, _, control_action_std = self._forward(x)
            control_action = torch.normal(control_action_mean, control_action_std)

            action = np.concatenate((scale_vector, control_action[0]))
            action = torch.tensor(action).unsqueeze(0)
            return action
        else:
            pass

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self._forward(x)
        if self.agent.env.stage == 'scale_transform':
            scale_state = actions[:, :self.scale_state_dim]
            return normal_log_density(scale_state, action_mean, action_log_std, action_std)
        elif self.agent.env.stage == 'execution':
            control_action = actions[:, self.scale_state_dim:]
            return normal_log_density(control_action, action_mean, action_log_std, action_std)
