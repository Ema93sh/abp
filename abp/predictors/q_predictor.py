import  logging
logger = logging.getLogger('root')

import tensorflow as tf
import numpy as np

from abp.adaptives.common.memory import Memory
from abp.adaptives.common.experience import Experience

from abp.models import DQNModel

class QPredictor(object):
    """ Predictor are equivalent to General Value Functions (GVFs) """

    #TODO
    # * discount factor how to decide?
    # * batch size should it be the same?
    # * save predictor

    def __init__(self, name, network_config, discount_factor = 0.99, batch_size = 32):
        super(QPredictor, self).__init__()
        self.name = name
        self.session = tf.Session()

        self.eval_model = DQNModel(name + "_eval", network_config, self.session)
        self.target_model = DQNModel(name + "_target", network_config, self.session)

        self.previous_state = None
        self.replay_memory = Memory(5000)
        self.discount_factor = discount_factor
        self.update_frequency = 1000
        self.batch_size = batch_size
        self.steps = 0

    def __del__(self):
        self.eval_model.save_network()
        self.target_model.save_network()
        self.session.close()

    def learn(self, current_state, action, reward, is_terminal, terminal_reward):
        self.steps += 1

        if is_terminal:
            reward = terminal_reward

        if action is None:
            action = 0

        if self.previous_state is not None:
            experience = Experience(self.previous_state, action, reward, current_state, is_terminal)
            self.replay_memory.add(experience)

        if self.steps % self.update_frequency == 0:
            logger.info("Predictor -- Replacing target model for %s" % self.name)
            self.target_model.replace(self.eval_model)

        self.previous_state = current_state

        self.update()

    def update(self):
        if self.replay_memory.current_size < self.batch_size:
            return

        batch = self.replay_memory.sample(self.batch_size)

        # TODO: Convert to tensor operations instead of for loops

        states = [experience.state for experience in batch]

        next_states = [experience.next_state for experience in batch]

        is_terminal = [ 0 if experience.is_terminal else 1 for experience in batch]

        actions = [experience.action for experience in batch]

        reward = [experience.reward for experience in batch]

        q_next = self.target_model.predict_batch(next_states)

        q_mean = np.mean(q_next, axis = 1)

        q_mean = np.array([ a * b if a == 0 else b for a,b in zip(is_terminal, q_mean)])

        q_values = self.eval_model.predict_batch(states)

        q_target = q_values.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = reward + self.discount_factor * q_mean

        self.eval_model.fit(states, q_target, self.steps)

    def predict(self, state):
        action, q_values = self.eval_model.predict(state)
        return q_values
