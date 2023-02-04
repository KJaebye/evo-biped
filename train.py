# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Training file
#   @author: Kangyao Huang
#   @created date: 23.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import logging
import torch
import numpy as np
from config.get_args import get_args
from config.config import Config
from utils.logger import Logger
from custom.agents.general_agent import GeneralAgent
from custom.agents.bipedalwalker_agent import BipedalWalkerAgent
from custom.agents.evo_bipedalwalker_agent import EvoBipedalWalkerAgent

if __name__ == "__main__":
    args = get_args()
    """ load env configs and training settings """
    cfg = Config(args.domain, args.task, rec=args.rec)

    """ set torch and cuda """
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) \
        if args.use_cuda and torch.cuda.is_available() else torch.device('cpu')
    # torch.cuda.is_available() is natively False on mac m1
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
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
    logger.critical('Type of current running: Training')
    logger.set_file_handler()

    start_iter = int(args.start_iter) if args.start_iter.isnumeric() else args.start_iter

    """ create agent """
    if cfg.domain == 'bipedalwalker':
        agent = BipedalWalkerAgent(args.task, args.domain, cfg, logger, dtype=dtype, device=device,
                         num_threads=args.num_threads, training=True, checkpoint=start_iter)
    elif cfg.domain == 'evo_bipedalwalker':
        agent = EvoBipedalWalkerAgent(args.task, args.domain, cfg, logger, dtype=dtype, device=device,
                         num_threads=args.num_threads, training=True, checkpoint=start_iter)
    else:
        agent = GeneralAgent(args.task, args.domain, cfg, logger, dtype=dtype, device=device,
                         num_threads=args.num_threads, training=True, checkpoint=start_iter)

    for iter in range(start_iter, start_iter + cfg.max_iter_num):
        agent.optimize(iter)
        # clean up GPU memory
        torch.cuda.empty_cache()
    agent.logger.critical('Training completed!')
