# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Training file
#   @author: Kangyao Huang
#   @created date: 23.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import os
import shutil

import logging
import torch
import argparse
import numpy as np
from lib.config.config import Config
from utils.logger import Logger
from custom.bipedalwalker.bipedalwalker_agent import BipedalWalkerAgent
from custom.evo_bipedalwalker.evo_bipedalwalker_agent import EvoBipedalWalkerAgent


def copyfile(srcfile, target_path):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(target_path):
            os.makedirs(target_path)  # 创建路径
        shutil.copy(srcfile, target_path + fname)  # 复制文件
        print("copy %s -> %s" % (srcfile, target_path + fname))


if __name__ == "__main__":
    # create a parser
    parser = argparse.ArgumentParser(description="Write in user's arguments from terminal.")

    # training configuration
    parser.add_argument('--domain', type=str, help='domain, must be specified to load the cfg file.', required=True)
    parser.add_argument('--task', type=str, help='task, must be specified to load the cfg file.', required=True)
    parser.add_argument('--algo', type=str, default='PPO2', help='algorithm to train the agent')
    parser.add_argument('--use_cuda', type=bool, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--rec', type=str, help='rec directory name')
    parser.add_argument('--start_iter', default='0')
    parser.add_argument('--num_threads', type=int, default=1)
    # parser.add_argument('--use_ggnn', type=bool, default=False, help='use NerveNet(GGNN) as policy networks')
    args = parser.parse_args()

    """ load envs configs and training settings """
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
        # create agent
        agent = BipedalWalkerAgent(args.task, args.domain, cfg, logger, dtype=dtype, device=device,
                                   num_threads=args.num_threads, training=True, checkpoint=start_iter, mean_action=True)

        # save .yml file in logging directory
        src_dir = './lib/config/cfg/bipedalwalker/'
        target_dir = logger.output_dir + '/'
        cfg_path = src_dir + cfg.task + '.yml'
        assert os.path.exists(cfg_path), 'This cfg file does not exist!'
        copyfile(cfg_path, target_dir)

    elif cfg.domain == 'evo_bipedalwalker':
        # create agent
        agent = EvoBipedalWalkerAgent(args.task, args.domain, cfg, logger, dtype=dtype, device=device,
                                      num_threads=args.num_threads, training=True, checkpoint=start_iter, mean_action=True)

        # save .yml file in logging directory
        src_dir = './lib/config/cfg/evo_bipedalwalker/'
        target_dir = logger.output_dir + '/'
        cfg_path = src_dir + cfg.task + '.yml'
        assert os.path.exists(cfg_path), 'This cfg file does not exist!'
        copyfile(cfg_path, target_dir)

    else:
        pass

    for iter in range(start_iter, start_iter + cfg.max_iter_num):
        agent.optimize(iter)
        # clean up GPU memory
        torch.cuda.empty_cache()
    agent.logger.critical('Training completed!')
