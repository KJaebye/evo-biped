import torch.nn as nn

class EvoBipedalWalkerPolicy(nn.Module):
    def __init__(self, cfg, agent):
        super(EvoBipedalWalkerPolicy, self).__init__()
        self.cfg = cfg
        self.agent = agent
        self.control_obs_dim = agent.env.obser

