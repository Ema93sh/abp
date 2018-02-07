import tensorflow as tf
import numpy as np

import operator
import os
import logging

from .aggregate_qtable import AggregateQTable

class DQAdaptive(object):
    """DQAdaptive using Q Learning algorithm with decomposed rewards"""

    def __init__(self, config):
        super(DQAdaptive, self).__init__()
        self.config = config
        self.learning = config.learning #config.True

        self.steps = 0
        self.previous_state = None
        self.previous_action = None
        self.current_reward = [0] * self.config.size_rewards #TODO: change reward into dictionary
        self.total_psuedo_reward = 0
        self.total_actual_reward = 0
        self.current_test_reward = 0 # Used once learning is disabled


        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        self.aggregate_qtable =  AggregateQTable(self.config.size_rewards, self.config.action_size, 0.001)

        dirname = os.path.join(config.job_dir, "tensorflow_summaries/%s/%s" %(config.name, "dqtable_summary"))
        run_number = 0 if not tf.gfile.IsDirectory(dirname) else len(tf.gfile.ListDirectory(dirname))
        self.writer = tf.summary.FileWriter("%s/%s" %(dirname, "run" + str(run_number)))

        if config.restore_model and self.config.model_path is not None:
            dirname = os.path.dirname(self.config.model_path)
            if tf.gfile.Exists(dirname):
                logging.info("Restoring model from %s" % self.config.model_path)
                self.aggregate_qtable.load(self.config.model_path)
            else:
                logging.error("Can't Restore model from %s the path does not exists" % self.config.model_path)

        self.episode = 0

    def __del__(self):
        self.session.close()
        self.writer.close()

    def save_model(self):
        if self.config.model_path is not None:
            dirname = os.path.dirname(self.config.model_path)
            if not tf.gfile.Exists(dirname):
                logging.info("Creating model path directories...")
                tf.gfile.MakeDirs(dirname)
            logging.info("Saving the model...")
            self.aggregate_qtable.save(self.config.model_path)


    def should_explore(self):
        epsilon = np.max([0.1, self.config.starting_epsilon * (self.config.epsilon_decay_rate ** (self.steps / self.config.decay_steps))])

        epsilon_summary = tf.Summary()
        epsilon_summary.value.add(tag='epsilon', simple_value = epsilon)
        self.writer.add_summary(epsilon_summary, self.steps)

        return np.random.choice([True, False],  p = [epsilon, 1 - epsilon])

    def predict(self, state):
        self.steps += 1

        if self.learning:
            self.update(state)

        if self.learning and self.should_explore():
            action = np.random.choice(self.config.action_size)
            q_values = None
        else:
            action, q_value = self.aggregate_qtable.qmax_merged(state)
            q_values = self.aggregate_qtable.get_for(state)

        self.current_reward = [0] * self.config.size_rewards

        self.previous_state = state
        self.previous_action = action

        return action, q_values

    def disable_learning(self):
        logging.info("Disabled Learning")
        self.learning = False
        self.episode = 0
        self.current_test_reward = 0
        self.save_model()

    def end_episode(self, state):
        if self.episode % 100 == 0:
            logging.info("End of Episode %d" % (self.episode + 1))

        self.episode += 1

        if self.learning:
            psuedo_reward_summary = tf.Summary()
            psuedo_reward_summary.value.add(tag='Total Psuedo rewards', simple_value = self.total_psuedo_reward)
            self.writer.add_summary(psuedo_reward_summary, self.episode)

            actual_reward_summary = tf.Summary()
            actual_reward_summary.value.add(tag='Total Actual rewards', simple_value = self.total_actual_reward)
            self.writer.add_summary(actual_reward_summary, self.episode)

            self.update(state, True)

            self.current_reward = [0] * self.config.size_rewards
            self.total_psuedo_reward = 0
            self.total_actual_reward = 0

            self.previous_state = None
            self.previous_action = None

        else:
            reward_summary = tf.Summary()
            reward_summary.value.add(tag='Test reward', simple_value = self.current_test_reward)
            self.writer.add_summary(reward_summary, self.episode)

            self.current_test_reward = 0

    def reward(self, r_type, r_value):
        self.total_psuedo_reward += r_value
        self.current_reward[r_type] += r_value

    def actual_reward(self, r): # Used to see the actual rewards while learning
        self.total_actual_reward += r

    def test_reward(self, r): # Used once learning is disabled
        self.current_test_reward += r

    def update(self, state, terminal = False):
        if self.previous_state is not None and self.previous_action is not None:
            next_action, q_value = self.aggregate_qtable.qmax_merged(state)

            for reward_type in range(self.config.size_rewards):
                prev_q = self.aggregate_qtable.get(reward_type, self.previous_state, self.previous_action)

                if not self.aggregate_qtable.contains(state) or terminal:
                    next_q = 0
                else:
                    next_q = self.aggregate_qtable.get(reward_type, state, next_action)

                updated_q = (1 - 0.01) * prev_q + 0.01 * (self.current_reward[reward_type] + self.config.gamma * next_q)

                self.aggregate_qtable.put(reward_type, self.previous_state, self.previous_action, updated_q)
        pass
