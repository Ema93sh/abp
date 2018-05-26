import logging
import time
import random
from copy import deepcopy

logger = logging.getLogger('root')


import numpy as np
import torch
from torch.autograd import Variable
from abp.adaptives.common.prioritized_memory.memory import PrioritizedReplayBuffer
from abp.utils import clear_summary_path
from abp.models import HRAModel
from tensorboardX import SummaryWriter

from baselines.common.schedules import LinearSchedule


class HRAAdaptive(object):
    """HRAAdaptive using HRA architecture"""

    def __init__(self, name, choices, reward_types, network_config, reinforce_config):
        super(HRAAdaptive, self).__init__()
        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.update_frequency = reinforce_config.update_frequency

        self.replay_memory = PrioritizedReplayBuffer(self.reinforce_config.memory_size, 0.6)
        self.learning = True
        self.explanation = False

        self.steps = 0
        self.previous_state = None
        self.previous_action = None
        self.reward_types = reward_types

        self.clear_rewards()


        self.total_reward = 0
        self.decomposed_total_reward = {}
        self.clear_episode_rewards()


        self.eval_model = HRAModel(self.name + "_eval", self.network_config)
        self.target_model = HRAModel(self.name + "_target", self.network_config)

        clear_summary_path(self.reinforce_config.summaries_path + "/" + self.name)
        self.summary = SummaryWriter(log_dir = self.reinforce_config.summaries_path + "/" + self.name)


        self.episode = 0
        self.beta_schedule = LinearSchedule(10 * 1000, initial_p = 0.2, final_p = 1.0)


    def __del__(self):
        self.summary.close()


    def should_explore(self):
        epsilon = np.max([0.1, self.reinforce_config.starting_epsilon * (
                         self.reinforce_config.decay_rate ** (self.steps / self.reinforce_config.decay_steps))])

        self.summary.add_scalar(tag='%s/Epsilon' % self.name, scalar_value=epsilon, global_step=self.steps)

        return  random.random() < epsilon


    def predict(self, state):
        self.steps += 1

        if self.previous_state is not None and self.previous_action is not None:
            self.replay_memory.add(self.previous_state, self.previous_action, self.reward_list(), state, False)

        if self.learning and self.should_explore():
            action = random.choice(list(range(len(self.choices))))
            q_values = [None] * len(self.choices)  # TODO should it be output shape or from choices?
            choice = self.choices[action]
        else:
            _state = Variable(torch.Tensor(state)).unsqueeze(0)
            predict_start_time =  time.time()
            action, q_values = self.eval_model.predict(_state)
            # print("predict time", time.time() - predict_start_time)

            choice = self.choices[action]

        if self.learning and self.steps % self.update_frequency == 0:
            logger.debug("Replacing target model for %s" % self.name)
            self.target_model.replace(self.eval_model)

        if self.steps % self.reinforce_config.update_steps == 0:
            update_start_time = time.time()
            self.update()
            # print("update time", time.time() - update_start_time)

        self.clear_rewards()

        self.previous_state = state
        self.previous_action = action

        return choice, q_values

    def disable_learning(self):
        logger.info("Disabled Learning for %s agent" % self.name)
        self.eval_model.save_network()
        self.target_model.save_network()

        self.learning = False
        self.episode = 0


    def end_episode(self, state):
        if not self.learning:
            return

        logger.info("End of Episode %d with total reward %.2f" % (self.episode + 1, self.total_reward))

        self.episode += 1

        self.summary.add_scalar(tag = '%s/Episode Reward' % self.name,
                                scalar_value = self.total_reward,
                                global_step = self.episode)

        for reward_type in self.reward_types:
            self.summary.add_scalar(tag = '%s/Decomposed Reward/%s' % (self.name, reward_type),
                                    scalar_value = self.decomposed_total_reward[reward_type],
                                    global_step = self.episode)

        self.replay_memory.add(self.previous_state, self.previous_action, self.reward_list(), state, True)

        self.clear_rewards()
        self.clear_episode_rewards()

        self.previous_state = None
        self.previous_action = None

        self.update()

    def reward_list(self):
        reward = [0] * len(self.reward_types)

        for i, reward_type in enumerate(sorted(self.reward_types)):
            reward[i] = self.current_reward[reward_type]

        return reward

    def clear_rewards(self):
        self.current_reward = {}
        for reward_type in self.reward_types:
            self.current_reward[reward_type] = 0

    def clear_episode_rewards(self):
        self.total_reward = 0
        self.decomposed_total_reward = {}
        for reward_type in self.reward_types:
            self.decomposed_total_reward[reward_type] = 0


    def reward(self, reward_type, value):
        self.current_reward[reward_type] += value
        self.decomposed_total_reward[reward_type] += value
        self.total_reward += value


    def update(self):
        if self.steps <= self.reinforce_config.batch_size:
            return

        beta = self.beta_schedule.value(self.steps)

        self.summary.add_scalar(tag='%s/Beta' % self.name, scalar_value=beta, global_step=self.steps)

        states, actions, reward, next_states, is_terminal, weights, batch_idxes = self.replay_memory.sample(self.reinforce_config.batch_size, beta)

        states = Variable(torch.Tensor(states))
        next_states = Variable(torch.Tensor(next_states))

        is_terminal = [0 if t else 1 for t in is_terminal]

        q_next = self.target_model.predict_batch(next_states)

        q_2 = np.mean(q_next, axis = 2)

        q_2 = is_terminal * q_2

        q_values = self.eval_model.predict_batch(states)

        q_target = q_values.copy()

        batch_index = np.arange(self.reinforce_config.batch_size, dtype=np.int32)

        q_target[:, batch_index, actions] = np.transpose(reward) + self.reinforce_config.discount_factor * q_2

        td_errors = q_values[:, batch_index, actions] - q_target[:, batch_index, actions]

        td_errors = td_errors.sum(axis=0)

        new_priorities = np.abs(td_errors) + 1e-6 #prioritized_replay_eps
        self.replay_memory.update_priorities(batch_idxes, new_priorities)

        self.eval_model.fit(states, q_target, self.steps)
        return td_errors
