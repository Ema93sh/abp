import logging

logger = logging.getLogger('root')

from copy import deepcopy
import numpy as np
import torch
from torch.autograd import Variable

from abp.adaptives.common.memory import Memory, Experience
from abp.utils import clear_summary_path
from abp.models import ActorModel
from tensorboardX import SummaryWriter

from baselines.common.schedules import LinearSchedule

# TODO Too many duplicate code. Need to refactor!

class PGAdaptive(object):
    """PGAdaptive using Vanilla Policy Gradient"""

    def __init__(self, name, choices, network_config, reinforce_config):
        super(PGAdaptive, self).__init__()
        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.update_frequency = reinforce_config.update_frequency

        self.replay_memory = Memory(self.reinforce_config.batch_size)

        self.steps = 0
        self.total_reward = 0

        self.previous_state = None
        self.previous_action = None
        self.clear_rewards()

        self.model = ActorModel(self.name + "_actor", self.network_config)
        self.summary = SummaryWriter(log_dir = self.reinforce_config.summaries_path + "/" + self.name)

        self.episode = 0
        self.epsilon_schedule = LinearSchedule(10 * 1000, initial_p = self.reinforce_config.starting_epsilon, final_p = 0.1)

    def __del__(self):
        self.summary.close()

    def predict(self, state):
        self.steps += 1


        if self.previous_state is not None and self.previous_action is not None:
            self.replay_memory.add((self.previous_state, self.previous_action, self.current_reward, state, False))

        _state = Variable(torch.Tensor(state)).unsqueeze(0)
        action_probs = self.model.predict(_state)

        #TODO continuous action
        m = Categorical(action_probs)
        action = m.sample()

        choice = self.choices[action]

        self.update()

        self.clear_rewards()

        self.previous_state = state
        self.previous_action = action

        return choice, q_values


    def disable_learning(self):
        logger.info("Disabled Learning for %s agent" % self.name)
        self.model.save_network()
        self.episode = 0


    def end_episode(self, state):
        if not self.learning:
            return

        logger.info("End of Episode %d with total reward %.2f" % (self.episode + 1, self.total_reward))

        self.episode += 1

        self.summary.add_scalar(tag='%s agent reward' % self.name,scalar_value=self.total_reward,
                                global_step=self.episode)

        self.replay_memory.add((self.previous_state, self.previous_action, self.reward_list(), state, True))

        self.clear_rewards()
        self.total_reward = 0

        self.previous_state = None
        self.previous_action = None

        self.update()

    def clear_rewards(self):
        self.current_reward = 0

    def reward(self, value):
        self.current_reward += value
        self.total_reward += value

    def update(self):
        if self.steps <= self.reinforce_config.batch_size:
            return

        states, actions, reward, next_states, is_terminal, weights, batch_idxes = self.replay_memory.sample(self.reinforce_config.batch_size,
                                                                                                                self.beta_schedule.value(self.steps))
        states = Variable(torch.Tensor(states))
        next_states = Variable(torch.Tensor(next_states))

        is_terminal = [0 if t else 1 for t in is_terminal]

        self.replay_memory.clear()

        self.model.fit(states, q_target, self.steps)

        return td_errors
