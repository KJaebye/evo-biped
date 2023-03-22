# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Evaluation file
#   @author: Kangyao Huang
#   @created date: 17.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

import logging
import torch
import numpy as np
import argparse

from lib.config.config import Config
from utils.logger import Logger
from custom.bipedalwalker.bipedalwalker_agent import BipedalWalkerAgent
from custom.evo_bipedalwalker.evo_bipedalwalker_agent import EvoBipedalWalkerAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, help='mujoco domain, must be specified to load the cfg file.',
                        required=True)
    parser.add_argument('--task', type=str, help='task, must be specified to load the cfg file.', required=True)
    parser.add_argument('--rec', type=str, help='rec directory name', required=True)
    parser.add_argument('--iter', default='best')
    parser.add_argument('--test', default=100)

    args = parser.parse_args()

    """ load envs configs and training settings """
    cfg = Config(args.domain, args.task, rec=args.rec)

    """ set torch and cuda """
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    """ logging config """
    # set logger
    logger = Logger(name='current', args=args, cfg=cfg)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # set output
    logger.set_output_handler()
    logger.print_system_info()

    # only training generates log file
    logger.critical('Type of current running: Evaluation. No log file will be created')

    iter = int(args.iter) if args.iter.isdigit() else args.iter

    """ create agent """
    if cfg.domain == 'bipedalwalker':
        agent = BipedalWalkerAgent(args.task, args.domain, cfg, logger, dtype=dtype, device=device,
                                      num_threads=1, training=False, checkpoint=iter)
    elif cfg.domain == 'evo_bipedalwalker':
        agent = EvoBipedalWalkerAgent(args.task, args.domain, cfg, logger, dtype=dtype, device=device,
                         num_threads=1, training=False, checkpoint=iter)
    else:
        pass

    agent.test()
