from memory import Memory
from model import DQNModel
from experience import Experience

import tensorflow as tf
import numpy as np
import os
import logging

class DQNAdaptive(object):
    """DQNAdaptive using DQN algorithm"""

    def __init__(self, config):
        super(DQNAdaptive, self).__init__()
        self.config = config
        self.replay_memory = Memory(config.memory_size)
        self.learning = config.learning #config.True

        self.steps = 0
        self.previous_state = None
        self.previous_action = None
        self.current_reward = 0
        self.total_psuedo_reward = 0
        self.total_actual_reward = 0
        self.current_test_reward = 0 # Used once learning is disabled

        # self.target_model = DQNModel(size_features, self.config.action_size, "target_model", trainable = False)
        self.eval_model = DQNModel(config.size_features, self.config.action_size, "eval_model")

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        dirname = os.path.join(config.job_dir, "tensorflow_summaries/%s/%s" %(config.name, "dqn_summary"))
        run_number = 0 if not tf.gfile.IsDirectory(dirname) else len(tf.gfile.ListDirectory(dirname))
        self.writer = tf.summary.FileWriter("%s/%s" %(dirname, "run" + str(run_number)), self.session.graph)

        # t_params = tf.get_collection('target_params')
        # e_params = tf.get_collection('eval_params')

        # self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if config.restore_model and self.config.model_path is not None:
            dirname = os.path.dirname(self.config.model_path)
            if tf.gfile.Exists(dirname) and len(tf.gfile.ListDirectory(dirname)) > 0:
                logging.info("Restoring model from %s" % self.config.model_path)
                self.saver.restore(self.session, self.config.model_path)
            else:
                logging.error("Cant Restore model from %s the path does not exists" % self.config.model_path)

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
            self.saver.save(self.session, self.config.model_path)

    def should_explore(self):
        epsilon = np.max([0.1, self.config.starting_epsilon * (self.config.epsilon_decay_rate ** (self.steps / self.config.decay_steps))])

        epsilon_summary = tf.Summary()
        epsilon_summary.value.add(tag='epsilon', simple_value = epsilon)
        self.writer.add_summary(epsilon_summary, self.steps)

        return np.random.choice([True, False],  p = [epsilon, 1 - epsilon])

    def predict(self, state):

        self.steps += 1

        # add to experience
        if self.previous_state is not None:
            experience = Experience(self.previous_state, self.previous_action, self.current_reward, state)
            self.replay_memory.add(experience)

        if self.learning and self.should_explore():
            action = np.random.choice(self.config.action_size)
        else:
            action = self.eval_model.predict(state, self.session)


        if self.learning and self.replay_memory.current_size > 30:
            self.update()

            # if self.steps % self.config.replace_target_steps == 0:
            #     self.session.run(self.replace_target_op)

            # if self.steps % 1000 == 0:
            #     self.target_model.generate_summaries(self.session, [state], self.writer, self.steps)

        self.current_reward = 0

        self.previous_state = state
        self.previous_action = action

        return action

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


            experience = Experience(self.previous_state, self.previous_action, self.current_reward, state, True)
            self.replay_memory.add(experience)

            self.current_reward = 0
            self.total_psuedo_reward = 0
            self.total_actual_reward = 0

            self.previous_state = None
            self.previous_action = None

            self.update()
        else:
            reward_summary = tf.Summary()
            reward_summary.value.add(tag='Test reward', simple_value = self.current_test_reward)
            self.writer.add_summary(reward_summary, self.episode)

            self.current_test_reward = 0

    def reward(self, r):
        self.total_psuedo_reward += r
        self.current_reward += r

    def actual_reward(self, r): # Used to see the actual rewards while learning
        self.total_actual_reward += r

    def test_reward(self, r): # Used once learning is disabled
        self.current_test_reward += r

    def update(self):
        batch_size = 32
        batch = self.replay_memory.sample(batch_size)

        batch_size = len(batch)

        # TODO: Convert to tensor operations instead of for loops

        states = [experience.state for experience in batch]

        next_states = [experience.next_state for experience in batch]

        is_terminal = [ 0 if experience.is_terminal else 1 for experience in batch]

        actions = [experience.action for experience in batch]

        reward = [experience.reward for experience in batch]

        q_next = self.eval_model.predict_batch(next_states, self.session)

        q_max = np.max(q_next, axis = 1)

        q_max = np.array([ a * b if a == 0 else b for a,b in zip(is_terminal, q_max)])

        q_values = self.eval_model.predict_batch(states, self.session)

        q_target = q_values.copy()

        batch_index = np.arange(batch_size, dtype=np.int32)

        q_target[batch_index, actions] = reward + self.config.gamma * q_max

        self.eval_model.fit(states, q_target, self.session, self.writer, self.steps)
