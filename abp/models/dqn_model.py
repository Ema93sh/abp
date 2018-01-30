import logging
logger = logging.getLogger('root')

import os

import tensorflow as tf
import numpy as np

from abp.utils import clear_summary_path

class DQNModel(object):
    """Neural Network for the DQN algorithm """

    def __init__(self, name, network_config, session, learning_rate = 0.001):
        super(DQNModel, self).__init__()
        self.name = name.replace(" ", "_")
        self.network_config = network_config
        self.collections = []

        # TODO add ability to configure learning rate for network!
        self.learning_rate = learning_rate

        self.summaries = []

        self.session = session

        self.build_network()

        self.saver = tf.train.Saver()

        self.session.run(tf.global_variables_initializer())

        # TODO
        # * Option to disable summaries

        clear_summary_path(self.network_config.summaries_path + "/" + self.name)

        self.summaries_writer = tf.summary.FileWriter(self.network_config.summaries_path + "/" + self.name)

        print "Created network for...", self.name

        self.restore_network()


    def __del__(self):
        self.summaries_writer.close()


    def save_network(self):
        if self.network_config.network_path and self.network_config.save_network:
            logger.info("Saving network for..." + self.name)
            dirname = os.path.dirname(self.network_config.network_path + "/" + self.name)
            if not tf.gfile.Exists(dirname):
                logger.info("Creating network path directories...")
                tf.gfile.MakeDirs(dirname)
            logger.info("Saving the network at %s" % self.network_config.network_path + "/" + self.name)
            self.saver.save(self.session, self.network_config.network_path + "/" + self.name)


    def restore_network(self):
        if self.network_config.restore_network and self.network_config.network_path:
            dirname = os.path.dirname(self.network_config.network_path + "/" + self.name)
            if not tf.gfile.Exists(dirname):
                logger.error("Can not restore model. Reason: The network path (%s) does not exists" % self.network_config.network_path)
                return
            self.saver.restore(self.session, self.network_config.network_path + "/" + self.name)


    def build_network(self):
        self.state = tf.placeholder(tf.float32, shape = [None, self.network_config.input_shape[0]], name = self.name + "_state")
        self.q_target = tf.placeholder(tf.float32, shape = [None, self.network_config.output_shape[0]], name = self.name + "_qvalues")

        w_initializer = tf.random_normal_initializer(0., 0.1)
        b_initializer = tf.constant_initializer(0.1)

        L = []

        with tf.variable_scope(self.name): #TODO. Create a valid scope name

            with tf.variable_scope("Hidden_Layer"):
                first_layer_size = self.network_config.layers[0]

                w = tf.get_variable("w1", shape = (self.network_config.input_shape[0], first_layer_size),
                                     initializer = w_initializer)

                b = tf.get_variable("b1", shape = (1, first_layer_size),
                                     initializer = b_initializer)

                L.append(tf.nn.relu(tf.matmul(self.state, w) + b))


                # Generate other Hidden Layers
                for i in range(1, len(self.network_config.layers)):
                    previous_layer_size = self.network_config.layers[i-1]
                    current_layer_size = self.network_config.layers[i]

                    w = tf.get_variable("w" + str(i+1),
                                    shape = (previous_layer_size, current_layer_size),
                                    initializer = w_initializer)

                    b = tf.get_variable("b" + str(i+1),
                                    shape = (1, current_layer_size),
                                    initializer = b_initializer)

                    l = tf.nn.relu(tf.matmul(L[i-1], w) + b)
                    L.append(l)

            with tf.variable_scope("Output_Layer"):
                w = tf.get_variable("w_output",
                                shape = (self.network_config.layers[-1], self.network_config.output_shape[0]),
                                initializer = w_initializer)

                b = tf.get_variable("b_output",
                                shape = (1, self.network_config.output_shape[0]),
                                initializer = b_initializer)

                self.q_current = tf.matmul(L[-1], w) + b

                self.summaries.append(tf.summary.histogram("Q", self.q_current))


        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_current))
            self.summaries.append(tf.summary.scalar('loss', self.loss))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


        self.merged_summary = tf.summary.merge(self.summaries)


    def predict(self, state):
        q_values = self.predict_batch([state])
        action = np.argmax(q_values[0])
        return action, q_values[0]


    def predict_batch(self, batch):
        q_values, = self.session.run([self.q_current], feed_dict = {self.state : batch})
        return q_values


    def generate_summaries(self, states, steps,  q_target = None):
        feed_dict = {self.state: states}

        if q_target is not None:
            feed_dict[self.q_target] = q_target

        summary_str = self.session.run(self.merged_summary, feed_dict = feed_dict)
        self.summaries_writer.add_summary(summary_str, steps)


    def fit(self, states, q_target, steps):
        if steps % 100 == 0:
            self.generate_summaries(states, steps, q_target)


        _  = self.session.run(self.train_op, feed_dict = {
                                                        self.state: states,
                                                        self.q_target: q_target
                                                      })

    def get_params(self):
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        return params

    def replace(self, from_model):
        t_params = self.get_params()
        f_params = from_model.get_params()
        self.session.run([ tf.assign(t, f) for t, f in zip(t_params, f_params) ])
