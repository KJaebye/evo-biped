# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class EvoBipedalWalkerAgent
#   @author: by Kangyao Huang
#   @created date: 02.Feb.2023
# ------------------------------------------------------------------------------------------------------------------- #


import os
import torch

import gym
import time
import pickle
import math
import multiprocessing
from lib.core.zfilter import ZFilter
from lib.utils.torch import *
from lib.agents.agent_ppo2 import AgentPPO2
from lib.core.memory import Memory
from custom.models.evo_bipedalwalker_policy import EvoBipedalWalkerPolicy
from custom.models.evo_bipedalwalker_critic import EvoBipedalWalkerValue

class Filter(ZFilter):
    def __init__(self, scale_state_dim, control_state_dim, demean=True, destd=True,
                 scale_clip=0.75, control_clip=10.0):
        self.fixed_dim = scale_state_dim
        self.control_clip = control_clip
        self.scale_clip = scale_clip
        super(Filter, self).__init__(control_state_dim, demean=demean, destd=destd, clip=control_clip)

    def __call__(self, x, update=True):
        x_fixed = x[:self.fixed_dim]
        x = x[self.fixed_dim:]

        if update and not self.fix:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.control_clip, self.control_clip)
        x_fixed = np.clip(x_fixed, 1 - self.scale_clip, 1 + self.scale_clip)
        x = np.concatenate((x_fixed, x))
        return x


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
        self.setup_env()

        super().__init__(self.cfg, self.env, self.logger, self.dtype, self.device, self.num_threads,
                         training=self.training, checkpoint=checkpoint, mean_action=mean_action)

        """ Check network variables are updated """
        self.scale_norm_n = []
        self.scale_norm_mean = []
        self.scale_norm_std = []
        self.last_scale_mlp0_weight = []
        self.last_scale_mlp0_bias = []
        self.last_scale_mlp1_weight = []
        self.last_scale_mlp1_bias = []
        self.scale_state_mean_weight = []
        self.scale_state_mean_bias = []

        self.control_norm_n = []
        self.control_norm_mean = []
        self.control_norm_std = []
        self.last_control_mlp0_weight = []
        self.last_control_mlp0_bias = []
        self.last_control_mlp1_weight = []
        self.last_control_mlp1_bias = []
        self.last_control_mlp2_weight = []
        self.last_control_mlp2_bias = []
        self.control_action_mean_weight = []
        self.control_action_mean_bias = []

    def setup_env(self):
        from custom.envs.evo_bipedalwalker import EvoBipedalWalker
        self.env = EvoBipedalWalker()
        self.env.seed()

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
        self.running_state = Filter(self.env.scale_state_dim, self.env.control_state_dim, scale_clip=1, control_clip=5)

        """define actor and critic"""
        self.policy_net = EvoBipedalWalkerPolicy(self.cfg, self)
        self.value_net = EvoBipedalWalkerValue(self.cfg, self)

        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)

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
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=self.mean_action)

        ################################################################################################################
        ##### Debug ####################################################################################################
        """ Check network variables are updated """
        # print(list(self.policy_net.named_modules()))

        # for name in self.policy_net.state_dict():
        #     print(name)
        #
        # print('scale_norm:')
        # print(self.policy_net.state_dict()['scale_norm.n'] == self.scale_norm_n)
        # print(self.policy_net.state_dict()['scale_norm.mean'] == self.scale_norm_mean)
        # print(self.policy_net.state_dict()['scale_norm.std'] == self.scale_norm_std)
        #
        # print('scale_mlp:')
        # print(self.policy_net.state_dict()['scale_mlp.affine_layers.0.weight'])
        # print(self.policy_net.state_dict()['scale_mlp.affine_layers.0.weight'] == self.last_scale_mlp0_weight)
        # print(self.policy_net.state_dict()['scale_mlp.affine_layers.0.bias'] == self.last_scale_mlp0_bias)
        # print(self.policy_net.state_dict()['scale_mlp.affine_layers.1.weight'] == self.last_scale_mlp1_weight)
        # print(self.policy_net.state_dict()['scale_mlp.affine_layers.1.bias'] == self.last_scale_mlp1_bias)
        # print('scale_state_mean:')
        # print(self.policy_net.state_dict()['scale_state_mean.weight'] == self.scale_state_mean_weight)
        # print(self.policy_net.state_dict()['scale_state_mean.bias'] == self.scale_state_mean_bias)

        # self.last_scale_mlp0_weight = self.policy_net.state_dict()['scale_mlp.affine_layers.0.weight']
        # self.last_scale_mlp0_bias = self.policy_net.state_dict()['scale_mlp.affine_layers.0.bias']
        # self.last_scale_mlp1_weight = self.policy_net.state_dict()['scale_mlp.affine_layers.1.weight']
        # self.last_scale_mlp1_bias = self.policy_net.state_dict()['scale_mlp.affine_layers.1.bias']
        # self.scale_state_mean_weight = self.policy_net.state_dict()['scale_state_mean.weight']
        # self.scale_state_mean_bias = self.policy_net.state_dict()['scale_state_mean.bias']

        # print('control_norm:')
        # print(self.policy_net.state_dict()['control_norm.n'] == self.control_norm_n)
        # print(self.policy_net.state_dict()['control_norm.mean'] == self.control_norm_mean)
        # print(self.policy_net.state_dict()['control_norm.std'] == self.control_norm_std)
        #
        # print('control_mlp:')
        # print(self.policy_net.state_dict()['control_mlp.affine_layers.0.weight'] == self.last_control_mlp0_weight)
        # print(self.policy_net.state_dict()['control_mlp.affine_layers.0.bias'] == self.last_control_mlp0_bias)
        # print(self.policy_net.state_dict()['control_mlp.affine_layers.1.weight'] == self.last_control_mlp1_weight)
        # print(self.policy_net.state_dict()['control_mlp.affine_layers.1.bias'] == self.last_control_mlp1_bias)
        # print(self.policy_net.state_dict()['control_mlp.affine_layers.2.weight'] == self.last_control_mlp2_weight)
        # print(self.policy_net.state_dict()['control_mlp.affine_layers.2.bias'] == self.last_control_mlp2_bias)
        #
        # print('control_action_mean:')
        # print(self.policy_net.state_dict()['control_action_mean.weight'] == self.control_action_mean_weight)
        # print(self.policy_net.state_dict()['control_action_mean.bias'] == self.control_action_mean_bias)

        ################################################################################################################
        ################################################################################################################

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


    def sample(self, min_batch_size, mean_action=False, render=False, training=True):
        def collect_samples(pid, queue, env, policy, custom_reward, mean_action, render, running_state, min_batch_size):
            if pid > 0:
                torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
                gym.utils.seeding.np_random(pid)
                # if hasattr(env, 'np_random'):
                #     env.np_random.integers(5000) * pid
                # env.np_random.seed(env.np_random.randint(5000) * pid)
                # if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
                #     env.env.np_random.seed(env.env.np_random.randint(5000) * pid)
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
                    # if mean_action:
                    #     action = policy(state_var)[0][0].detach().numpy()
                    # else:
                    #     action = policy.select_action(state_var)[0].detach().numpy()

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
                        """ post step """
                        def post_step():
                            # initial input scale state
                            input_scale_state = policy.input_scale_state
                            policy.input_scale_state.requires_grad_(True)
                            observation, _ = env.execution_reset()
                            state = np.concatenate((input_scale_state[0].detach().numpy(), observation))

                            # scale state that used in this episode
                            env.stage = 'scale_transform'
                            # print(input_scale_state)
                            scale_state_mean, _, scale_std = policy._forward(input_scale_state)
                            # print(scale_state_mean)
                            assert scale_state_mean.all() == policy.agent.env.scale_vector.all()
                            control_action_padding = torch.zeros([1, 4], dtype=torch.float)
                            action = torch.cat((scale_state_mean, control_action_padding), -1)
                            action = action[0].detach().numpy()

                            # mask
                            mask = 0
                            # next state (has no meaning)
                            next_state, _, _, _, _ = env.step(action)
                            # total reward during this episode using current morphology
                            reward = reward_episode
                            # print(reward)
                            return state, action, mask, next_state, reward

                        state, action, mask, next_state, reward = post_step()
                        memory.push(state, action, mask, next_state, reward)
                        break

                    # rendering
                    # env.render()

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

        """ Multiprocessing sampling """
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
