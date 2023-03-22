# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class EvoBipedalWalkerAgent
#   @author: by Kangyao Huang
#   @created date: 02.Feb.2023
# ------------------------------------------------------------------------------------------------------------------- #


import os

import gym
import time
import pickle
import math
import multiprocessing
import random
from lib.utils.torch import *
from lib.core.trajbatch import TrajBatchDisc
from lib.agents.agent_ppo2 import AgentPPO2
from lib.core.common import estimate_advantages
from custom.evo_bipedalwalker.evo_bipedalwalker_policy import EvoBipedalWalkerPolicy
from custom.evo_bipedalwalker.evo_bipedalwalker_critic import EvoBipedalWalkerValue


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a tuple."""
        self.memory.append([*args])

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.memory
        else:
            random_batch = random.sample(self.memory, batch_size)
            return random_batch

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


class EvoBipedalWalkerAgent(AgentPPO2):
    def __init__(self, task, domain, cfg, logger, dtype, device, num_threads, training=True, checkpoint=0,
                 mean_action=False):
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint
        self.setup_env(task)
        self.setup_networks()

        super().__init__(self.cfg, self.env, self.logger, self.dtype, self.device, self.num_threads,
                         training=self.training, checkpoint=checkpoint, mean_action=mean_action)


    def setup_env(self, task):
        from custom.evo_bipedalwalker.evo_bipedalwalker import EvoBipedalWalker
        from custom.evo_bipedalwalker.evo_bipedalwalker import EvoBipedalWalkerHardcore
        if task == 'easy':
            self.env = EvoBipedalWalker(self.logger)
        else:
            self.env = EvoBipedalWalkerHardcore(self.logger)
        self.env.seed(self.cfg.seed)

        # dimension define
        self.stage_state_dim = 1
        self.scale_state_dim = self.env.scale_state_dim
        self.sim_obs_dim = self.env.sim_obs_dim
        self.state_dim = self.stage_state_dim + self.scale_state_dim + self.sim_obs_dim

        self.sim_action_dim = self.env.sim_action_dim
        self.action_dim = self.scale_state_dim + self.sim_action_dim
        self.running_state = None

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

        """ define actor and critic """
        self.policy_net = EvoBipedalWalkerPolicy(self.cfg, self)
        self.value_net = EvoBipedalWalkerValue(self.cfg, self)

        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)

        self.sample_modules = [self.policy_net]
        self.update_modules = [self.policy_net, self.value_net]

    def save_checkpoint(self, iter, log, log_eval):
        def save(checkpoint_path):
            to_device(torch.device('cpu'), self.policy_net, self.value_net)
            model_checkpoint = \
                {
                    'policy_dict': self.policy_net.state_dict(),
                    'value_dict': self.value_net.state_dict(),
                    'running_state': self.running_state,
                    'best_reward': self.best_reward,
                    'iter': iter
                }
            pickle.dump(model_checkpoint, open(checkpoint_path, 'wb'))
            to_device(self.device, self.policy_net, self.value_net)

        cfg = self.cfg

        if cfg.save_model_interval > 0 and (iter + 1) % cfg.save_model_interval == 0:
            self.tb_logger.flush()
            self.logger.critical(f'Saving the interval checkpoint with rewards {self.best_reward:.2f}')
            save('%s/iter_%04d.p' % (cfg.model_dir, iter + 1))

        if log_eval['avg_reward'] > self.best_reward:
            self.best_reward = log_eval['avg_reward']
            self.save_best_flag = True
            self.logger.critical('Get the best episode reward: {:.2f}'.format(self.best_reward))

        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.critical(f'Saving the best checkpoint with rewards {self.best_reward:.2f}')
            save('%s/best.p' % self.cfg.model_dir)
            self.save_best_flag = False

    def optimize(self, iter):
        """
        Optimize and main part of logging.
        """
        self.logger.info(
            '#-------------------------------- Iteration {} ----------------------------------#'.format(iter))

        """ generate multiple trajectories that reach the minimum batch_size """
        t0 = time.time()
        batch, log = self.sample(self.cfg.batch_size)
        t1 = time.time()
        self.logger.info('Sampling time: {:.2f} s by {} slaves'.format(t1 - t0, self.num_threads))
        self.update_params(batch, iter)
        t2 = time.time()
        self.logger.info('Policy update time: {:.2f} s'.format(t2 - t1))

        """ evaluate with determinstic action (remove noise for exploration) """
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=self.mean_action, training=False)


        """ logging """
        self.tb_logger.add_scalar('train_R_avg', log['avg_reward'], iter)
        self.tb_logger.add_scalar('eval_R_eps_avg', log_eval['avg_reward'], iter)

        self.logger.info('Average TRAINING episode reward: {:.2f}'.format(log['avg_reward']))
        self.logger.info('Maximum TRAINING episode reward: {:.2f}'.format(log['max_reward']))
        self.logger.info('Average EVALUATION episode reward: {:.2f}'.format(log_eval['avg_reward']))
        self.save_checkpoint(iter, log, log_eval)
        t_cur = time.time()
        self.logger.info('Total time: {:10.2f} min'.format((t_cur - self.t_start) / 60))
        self.total_steps += self.cfg.batch_size
        self.logger.info('{} total steps have happened'.format(self.total_steps))

    def collect_samples(self, pid, queue, env, policy_net, custom_reward, mean_action, render, running_state, min_batch_size):
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
            num_episodes += 1
            obs, info = env.reset()
            state = obs

            if running_state is not None:
                state = running_state(state)
            reward_episode = 0

            for t in range(10000):
                state_var = tensorfy([state])
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                action = policy_net.select_action(state_var, use_mean_action).detach().numpy()[0].astype(np.float64)
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = obs
                # print(obs)

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
                    if self.cfg.use_post_reward:
                        """ post step """
                        # using no_grad() option since auto_grad is not allowed in multiprocessing
                        with torch.no_grad():
                            scale_dist, _, _, _ = policy_net.forward(policy_net.fixed_x)
                            expected_scale_vector = scale_dist.mean_sample()
                            action = torch.zeros([1, self.action_dim], dtype=torch.float)
                            action[:, :self.scale_state_dim] = expected_scale_vector
                            action = action.detach().numpy()[0].astype(np.float64)

                        # mask
                        mask = 0

                        # get current state and next state (has no meaning)
                        stage_ind = policy_net.fixed_x[0][0].numpy()
                        scale_state = policy_net.fixed_x[0][1].numpy()
                        sim_obs = policy_net.fixed_x[0][2].numpy()
                        state = [stage_ind, scale_state, sim_obs]
                        next_state = [stage_ind, expected_scale_vector.numpy(), sim_obs]

                        # total reward
                        reward = reward_episode
                        memory.push(state, action, mask, next_state, reward)
                    break

                # rendering
                # env.render()

                state = next_state

            env.close()

            # log stats
            num_steps += (t + 1)
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

    def merge_log(self, log_list):
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

    def sample(self, min_batch_size, mean_action=False, render=False, training=True):
        """ Multiprocessing sampling """
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy_net)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        slaves = []
        memories = [None] * self.num_threads
        logs = [None] * self.num_threads

        for i in range(self.num_threads - 1):
            slave_args = (i + 1, queue, self.env, self.policy_net, self.custom_reward, mean_action,
                           False, self.running_state, thread_batch_size)
            slaves.append(multiprocessing.Process(target=self.collect_samples, args=slave_args))
        for slave in slaves:
            slave.start()

        memories[0], logs[0] = self.collect_samples(0, None, self.env, self.policy_net, self.custom_reward, mean_action,
                                      render, self.running_state, thread_batch_size)

        for i in range(self.num_threads-1):
            pid, slave_memory, slave_log = queue.get()
            memories[pid] = slave_memory
            logs[pid] = slave_log

        traj_batch = TrajBatchDisc(memories)

        log = self.merge_log(logs)
        to_device(self.device, self.policy_net)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        return traj_batch, log


    def update_params(self, batch, iter):
        to_train(*self.update_modules)
        states = tensorfy(batch.states, self.device)
        actions = tensorfy(batch.actions, self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)

        if self.cfg.use_post_reward:
            """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            """ For post reward step """""""""""""""""""""""""""""""""""""""""
            scale_index = []
            for i, states_i in enumerate(states):
                # print(states_i[0])
                if states_i[0] == 0:
                    scale_index.append(i)
            states_scale = [states[i] for i in scale_index]
            for i in reversed(scale_index): states.pop(i)
            actions_scale = [actions[i] for i in scale_index]
            for i in reversed(scale_index): actions.pop(i)
            rewards_scale = torch.tensor([rewards[i] for i in scale_index])
            for i in reversed(scale_index): rewards = rewards[torch.arange(rewards.size(0)) != i]
            masks_scale = torch.tensor([masks[i] for i in scale_index])
            for i in reversed(scale_index): masks = masks[torch.arange(masks.size(0)) != i]

            with torch.no_grad():
                values_scale = self.value_net(states_scale)
                fixed_log_probs_scale = self.policy_net.get_log_prob(states_scale, actions_scale)

            advantages_scale, returns_scale = estimate_advantages(rewards_scale, masks_scale, values_scale,
                                                                  self.cfg.gamma, self.cfg.tau, self.device)

            # update morphology network
            self.ppo_step(self.policy_net, self.value_net, self.optimizer_policy, self.optimizer_value,
                          1, states_scale, actions_scale, returns_scale, advantages_scale, fixed_log_probs_scale,
                          self.cfg.clip_epsilon, self.cfg.l2_reg, iter)

            # choose to freeze the control network when updating scale network
            # def freeze(fc):
            #     fc.requires_grad = False
            #
            # def defreeze(fc):
            #     fc.requires_grad = True
            #
            # # freeze the control network
            # freeze(self.policy_net.control_norm)
            # freeze(self.policy_net.control_mlp)
            # freeze(self.policy_net.control_action_mean)
            # freeze(self.policy_net.control_action_log_std)
            #
            # # update morphology network
            # self.ppo_step(self.policy_net, self.value_net, self.optimizer_policy, self.optimizer_value,
            #               1, states_scale, actions_scale, returns_scale, advantages_scale, fixed_log_probs_scale,
            #               self.cfg.clip_epsilon, self.cfg.l2_reg, iter)
            #
            # # defreeze the control network
            # defreeze(self.policy_net.control_norm)
            # defreeze(self.policy_net.control_mlp)
            # defreeze(self.policy_net.control_action_mean)
            # defreeze(self.policy_net.control_action_log_std)

        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.cfg.gamma, self.cfg.tau, self.device)

        """perform mini-batch PPO update"""
        self.logger.info('| %16s | %16s | %16s |' % ('policy_loss', 'value_loss', 'entropy'))
        optim_iter_num = int(math.ceil(len(states) / self.cfg.mini_batch_size))
        for _ in range(self.cfg.optim_num_epoch):
            perm_np = np.arange(len(states))
            np.random.shuffle(perm_np)
            perm = torch.LongTensor(perm_np).to(self.device)

            def index_select_list(x, ind):
                return [x[i] for i in ind]

            states, actions, returns, advantages, fixed_log_probs = \
                index_select_list(states, perm_np), index_select_list(actions, perm_np), \
                returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()

            policy_loss, value_loss, entropy = [], [], []
            for i in range(optim_iter_num):
                ind = slice(i * self.cfg.mini_batch_size, min((i + 1) * self.cfg.mini_batch_size, len(states)))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                policy_loss_i, value_loss_i, entropy_i = \
                    self.ppo_step(self.policy_net, self.value_net, self.optimizer_policy, self.optimizer_value,
                                  1, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b,
                                  self.cfg.clip_epsilon, self.cfg.l2_reg, iter)

                policy_loss.append(policy_loss_i.detach().numpy())
                value_loss.append(value_loss_i.detach().numpy())
                entropy.append(entropy_i.detach().numpy())
            self.logger.info('| %16.4f | %16.4f | %16.4f |' %
                             (np.mean(policy_loss), np.mean(value_loss), np.mean(entropy)))