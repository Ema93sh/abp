import logging
logger = logging.getLogger('root')

import time
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from baselines.common.schedules import LinearSchedule

from abp.utils import clear_summary_path
from abp.models import DQNModel
from abp.adaptives.common.prioritized_memory.memory import PrioritizedReplayBuffer

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



class DQNAdaptive(object):
    """Adaptive which uses the  DQN algorithm"""

    def __init__(self, name, choices, network_config, reinforce_config):
        super(DQNAdaptive, self).__init__()
        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config

        self.replay_memory = PrioritizedReplayBuffer(self.reinforce_config.memory_size, 0.6)
        self.learning = True

        self.steps = 0
        self.previous_state = None
        self.previous_action = None
        self.current_reward = 0
        self.total_reward = 0

        clear_summary_path(self.reinforce_config.summaries_path + "/" + self.name)
        self.summary = SummaryWriter(log_dir = self.reinforce_config.summaries_path + "/" + self.name)

        self.target_model = DQNModel(self.name + "_target", self.network_config)
        self.eval_model = DQNModel(self.name + "_eval", self.network_config)

        self.episode = 0
        self.beta_schedule = LinearSchedule(10 * 1000, initial_p = 0.2, final_p = 1.0)

    def should_explore(self):
        epsilon = np.max([0.1, self.reinforce_config.starting_epsilon * (self.reinforce_config.decay_rate ** (self.steps / self.reinforce_config.decay_steps))])
        self.summary.add_scalar(tag='epsilon', scalar_value=epsilon, global_step=self.steps)

        return random.random() < epsilon

    def predict(self, state):
        self.steps += 1
        saliencies = []

        # add to experience
        if self.previous_state is not None:
            experience = (self.previous_state, self.previous_action, self.current_reward, state, False)
            self.replay_memory.add(*experience)

        if self.learning and self.should_explore():
            q_values = None
            choice = random.choice(self.choices)
            action = self.choices.index(choice)
        else:
            _state = torch.Tensor(state).unsqueeze(0)
            action, q_values = self.eval_model.predict(_state, self.steps)
            choice = self.choices[action]

        if self.learning and self.steps % self.reinforce_config.replace_frequency == 0:
            logger.debug("Replacing target model for %s" % self.name)
            self.target_model.replace(self.eval_model)

        if self.learning and self.steps % self.reinforce_config.update_steps == 0:
            self.update()

        self.current_reward = 0
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

        logger.info("End of Episode %d with total reward %d" % (self.episode + 1, self.total_reward))

        self.episode += 1
        self.summary.add_scalar(tag = '%s/Episode Reward' % self.name,
                                scalar_value = self.total_reward,
                                global_step = self.episode)

        experience = (self.previous_state, self.previous_action, self.current_reward, state, True)
        self.replay_memory.add(*experience)


        if self.episode % self.network_config.save_steps == 0:
            self.eval_model.save_network()
            self.target_model.save_network()

        self.reset()


    def reset(self):
        self.current_reward = 0
        self.total_reward = 0
        self.previous_state = None
        self.previous_action = None


    def reward(self, r):
        self.total_reward += r
        self.current_reward += r


    def update(self):
        if self.steps <= self.reinforce_config.batch_size:
            return

        beta = self.beta_schedule.value(self.steps)
        self.summary.add_scalar(tag='%s/Beta' % self.name, scalar_value=beta, global_step=self.steps)

        states, actions, reward, next_states, is_terminal, weights, batch_idxes = self.replay_memory.sample(self.reinforce_config.batch_size, beta)

        states = FloatTensor(states)
        next_states = FloatTensor(next_states)
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.reinforce_config.batch_size, dtype = torch.long)

        #Current Q Values
        q_actions, q_values  = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]


        #Calculate target
        actions, q_next = self.target_model.predict_batch(next_states)
        q_max = q_next.max(1)[0].detach()
        q_max = (1 - terminal) * q_max

        q_target = reward + self.reinforce_config.discount_factor * q_max


        #update model
        self.eval_model.fit(q_values, q_target, self.steps)

        #Update priorities
        td_errors = q_values - q_target
        new_priorities = torch.abs(td_errors) + 1e-6 #prioritized_replay_eps
        self.replay_memory.update_priorities(batch_idxes, new_priorities.data)
