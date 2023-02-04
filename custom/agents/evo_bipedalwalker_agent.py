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
        self.running_state = ZFilter((self.env.observation_space.shape[0]), clip=5)
        cfg = self.cfg

        """define actor and critic"""
        self.policy_net = EvoBipedalWalkerPolicy()
        self.value_net = EvoBipedalWalkerValue()

        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)


    def sample(self, min_batch_size, mean_action=False, render=False):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        slaves = []

        for i in range(self.num_threads - 1):
            slave_args = (i + 1, queue, self.env, self.policy, self.custom_reward, mean_action,
                           False, self.running_state, thread_batch_size)
            slaves.append(multiprocessing.Process(target=collect_samples, args=slave_args))
        for slave in slaves:
            slave.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, mean_action,
                                      render, self.running_state, thread_batch_size)

        slave_logs = [None] * len(slaves)
        slave_memories = [None] * len(slaves)

        for _ in slaves:
            pid, slave_memory, slave_log = queue.get()
            slave_memories[pid - 1] = slave_memory
            slave_logs[pid - 1] = slave_log

        for slave_memory in slave_memories:
            memory.append(slave_memory)

        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + slave_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log


def collect_samples(pid, queue, env, policy, custom_reward, mean_action, render, running_state, min_batch_size):
    if pid > 0:
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        gym.utils.seeding.np_random(pid)

    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    while num_steps < min_batch_size:
        observation, info = env.reset()
        state = observation

        if running_state is not None:
            state = running_state(state)
        reward_episode = 0

        for t in range(10000):
            state_var = torch.tensor(state).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)

            observation, reward, terminated, truncated, info = env.step(action)
            next_state = observation

            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if terminated else 1

            memory.push(state, action, mask, next_state, reward)

            if terminated or truncated:
                break

            state = next_state

        env.close()

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log