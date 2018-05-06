import logging
logger = logging.getLogger('root')

import tensorflow as tf
import numpy as np

from abp.adaptives.common.memory import Memory
from abp.adaptives.common.experience import Experience
from abp.utils import clear_summary_path
from abp.models import CriticModel, ActorModel

#TODO Too many duplicate code. Need to refactor!

class A3CAdaptive(object):
    """A3CAdaptive using Actor Critic Algorithm"""
    def __init__(self, name, choices, network_config, reinforce_config):
        super(A3CAdaptive, self).__init__()
        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.update_frequency = reinforce_config.update_frequency

        self.replay_memory = Memory(self.reinforce_config.memory_size)
        self.learning = True

        self.steps = 0
        self.previous_state = None
        self.previous_action = None
        self.reward_types = len(self.network_config.networks)
        self.current_reward = 0
        self.total_reward = 0
        self.session = tf.Session()

        self.critic_model = CriticModel(self.name + "_critic", self.network_config, self.session)
        self.actor_model = ActorModel(self.name + "_actor", self.network_config, self.session)

        #TODO:
        # * Add more information/summaries related to reinforcement learning
        # * Option to disable summary?
        clear_summary_path(self.reinforce_config.summaries_path + "/" + self.name)

        self.summaries_writer = tf.summary.FileWriter(self.reinforce_config.summaries_path + "/" + self.name, graph = self.session.graph)

        self.episode = 0

    def __del__(self):
        self.summaries_writer.close()
        self.session.close()

    def should_explore(self):
        epsilon = np.max([0.1, self.reinforce_config.starting_epsilon * (self.reinforce_config.decay_rate ** (self.steps / self.reinforce_config.decay_steps))])

        epsilon_summary = tf.Summary()
        epsilon_summary.value.add(tag='epsilon', simple_value = epsilon)
        self.summaries_writer.add_summary(epsilon_summary, self.steps)

        return np.random.choice([True, False],  p = [epsilon, 1 - epsilon])


    def predict(self, state):
        self.steps += 1

        if self.learning:
            # TODO add noise when learning is True
            actor_prob = self.actor_model.predict(state)
            critic_values = self.critic_model.predict(state)
            action = np.random.choice(range(len(self.choices)), p = actor_prob)
            choice = self.choices[action]
        else:
            actor_prob = self.actor_model.predict(state)
            critic_values = self.critic_model.predict(state)
            action = np.random.choice(range(len(self.choices)), p = actor_prob)
            choice = self.choices[action]

        # add to experience
        if self.previous_state is not None and self.previous_action is not None:
            experience = Experience(self.previous_state, self.previous_action, self.current_reward, state, action)
            self.replay_memory.add(experience)



        # TODO
        # if self.learning and self.steps % self.update_frequency == 0:
        #     logger.debug("Replacing target model for %s" % self.name)
        #     self.target_model.replace(self.eval_model)

        self.update()

        self.current_reward = 0

        self.previous_state = state
        self.previous_action = action

        return choice, actor_prob, critic_values

    def disable_learning(self):
        logger.info("Disabled Learning for %s agent" % self.name)
        self.actor_model.save_network()
        self.critic_model.save_network()

        self.learning = False
        self.episode = 0

    def end_episode(self, state):
        if not self.learning:
            return

        logger.info("End of Episode %d with total reward %d" % (self.episode + 1, self.total_reward))

        self.episode += 1

        reward_summary = tf.Summary()
        reward_summary.value.add(tag='%s agent reward' % self.name, simple_value = self.total_reward)
        self.summaries_writer.add_summary(reward_summary, self.episode)

        experience = Experience(self.previous_state, self.previous_action, self.current_reward, state,  is_terminal = True)
        self.replay_memory.add(experience)

        self.current_reward = 0
        self.total_reward = 0

        self.previous_state = None
        self.previous_action = None

        self.update()

    def reward(self, reward):
        self.total_reward += reward
        for i in range(self.reward_types):
            self.current_reward[i] += decomposed_rewards[i]


    def update_critic(self, batch):
        # TODO: Convert to tensor operations instead of for loops

        states = [experience.state for experience in batch]

        next_states = [experience.next_state for experience in batch]

        is_terminal = np.array([ 0 if experience.is_terminal else 1 for experience in batch])

        actions = [experience.action for experience in batch]

        reward = np.array([experience.reward for experience in batch])

        v_next = self.critic_model.predict_batch(next_states)

        v_next = is_terminal.reshape(self.reinforce_config.batch_size, 1) * v_next

        v_current = self.critic_model.predict_batch(states)

        v_target = reward.reshape(self.reinforce_config.batch_size, 1) + self.reinforce_config.discount_factor * v_next

        self.critic_model.fit(states, v_target, self.steps)


    def update_actor(self, batch):
        states = [experience.state for experience in batch]

        is_terminal = np.array([ 0 if experience.is_terminal else 1 for experience in batch])

        actions = np.array([experience.action for experience in batch]).reshape(self.reinforce_config.batch_size, 1)

        v_current = is_terminal.reshape(self.reinforce_config.batch_size, 1) * self.critic_model.predict_batch(states)

        self.actor_model.fit(states, actions, v_current, self.steps)


    def update(self):
        if self.replay_memory.current_size < self.reinforce_config.batch_size:
            return

        batch = self.replay_memory.sample(self.reinforce_config.batch_size)

        self.update_critic(batch)

        self.update_actor(batch)
