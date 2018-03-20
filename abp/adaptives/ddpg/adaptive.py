import logging
logger = logging.getLogger('root')

import tensorflow as tf
import numpy as np
import time

from abp.adaptives.common.memory import Memory
from abp.utils import clear_summary_path


class DDPGAdaptive(object):
    """ Adaptive which uses the Deep Deterministic Policy Gradient(DDPG)  """

    def __init__(self, name, choices, network_config, reinforce_config):
        super(DDPGAdaptive, self).__init__()
        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config

        self.learning = True

        self.memory = Memory(self.reinforce_config.memory_size)

        self.steps = 0
        self.previous_state = None
        self.previous_action = None
        self.current_reward = 0
        self.total_reward = 0
        self.session = tf.Session()

        self.model = PolicyModel(self.name, self.network_config, self.session)

        #TODO:
        # * Add more information/summaries related to reinforcement learning
        # * Option to diable summary?
        clear_summary_path(self.reinforce_config.summaries_path + "/" + self.name)

        #TODO
        self.summaries_writer = tf.summary.FileWriter(self.reinforce_config.summaries_path + "/" + self.name, graph = self.session.graph)

        self.episode = 0


    def __del__(self):
        self.summaries_writer.close()
        self.session.close()


    def predict(self, state):
        self.steps += 1

        # add to experience
        experience = Experience(self.previous_state, self.previous_action, self.current_reward, state)
        self.memory.add(experience)

        action, probs = self.model.predict(state)
        choice = self.choices[action]

        self.current_reward = 0

        self.previous_state = state
        self.previous_action = action

        return choice, probs


    def disable_learning(self):
        logger.info("Disabled Learning for %s agent" % self.name)
        self.model.save_network()

        self.learning = False
        self.episode = 0


    def end_episode(self, state):
        self.episode += 1

        logger.info("End of Episode %d with total reward %d" % (self.episode + 1, self.total_reward))

        if not self.learning:
            return

        reward_summary = tf.Summary()
        reward_summary.value.add(tag='%s agent reward' % self.name, simple_value = self.total_reward)
        self.summaries_writer.add_summary(reward_summary, self.episode)

        self.current_reward = 0
        self.total_reward = 0

        self.previous_state = None
        self.previous_action = None


    def reward(self, r):
        self.total_reward += r
        self.current_reward += r


    def update(self):
        # TODO: Convert to tensor operations instead of for loops
        experiences = self.memory.all()

        states = [experience.state for experience in experiences]

        next_states = [experience.next_state for experience in experiences]

        is_terminal = [ 0 if experience.is_terminal else 1 for experience in experiences]

        actions = [experience.action for experience in experiences]

        reward = [experience.reward for experience in experiences]

        self.model.fit(states, actions, reward, next_states, is_terminal)
