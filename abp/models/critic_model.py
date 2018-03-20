import logging
logger = logging.getLogger('root')

import os

import tensorflow as tf
import numpy as np

from abp.utils import clear_summary_path

class CriticModel(object):
    """Neural Network for the Critic """

    def __init__(self, name, network_config, session, restore = True, learning_rate = 0.001):
        super(CriticModel, self).__init__()
        self.name = name.replace(" ", "_")
        self.network_config = network_config
        self.collections = []
        self.restore = restore

        # TODO add ability to configure learning rate for network!
        self.learning_rate = learning_rate

        self.summaries = []

        self.session = session

        logger.info("Building network for %s" % self.name)

        self.build_network()

        self.saver = tf.train.Saver()

        self.session.run(tf.global_variables_initializer())

        # TODO
        # * Option to disable summaries

        clear_summary_path(self.network_config.summaries_path + "/" + self.name)

        self.summaries_writer = tf.summary.FileWriter(self.network_config.summaries_path + "/" + self.name)

        logger.info("Created network for %s " % self.name)

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
        if self.restore and self.network_config.restore_network and self.network_config.network_path:
            dirname = os.path.dirname(self.network_config.network_path + "/" + self.name)
            if not tf.gfile.Exists(dirname):
                logger.error("Can not restore model. Reason: The network path (%s) does not exists" % self.network_config.network_path)
                return
            logger.info("Restoring network for %s " % self.name)
            self.saver.restore(self.session, self.network_config.network_path + "/" + self.name)


    def build_network(self):
        self.state = tf.placeholder(tf.float32, shape = [None] + self.network_config.input_shape, name = self.name + "_state")
        self.value_target = tf.placeholder(tf.float32, shape = [None, 1], name = self.name + "_values")

        w_initializer = tf.random_normal_initializer(0., 0.1)
        b_initializer = tf.constant_initializer(0.1)

        L = []

        with tf.variable_scope(self.name + "_critic"): #TODO. Create a valid scope name

            shape = self.state.get_shape().as_list()        # a list: [None, 9, 2]
            input_layer_size = np.prod(shape[1:])           # input_layer_size = prod(9,2) = 18
            input_layer = tf.reshape(self.state, [-1, input_layer_size])

            with tf.variable_scope("Hidden_Layer"):
                first_layer_size = self.network_config.layers[0]

                w = tf.get_variable("w1", shape = (input_layer_size, first_layer_size),
                                     initializer = w_initializer)

                b = tf.get_variable("b1", shape = (1, first_layer_size),
                                     initializer = b_initializer)

                L.append(tf.nn.relu(tf.matmul(input_layer, w) + b))


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
                                shape = (self.network_config.layers[-1], 1),
                                initializer = w_initializer)

                b = tf.get_variable("b_output",
                                shape = (1, 1),
                                initializer = b_initializer)

                self.value_current = tf.matmul(L[-1], w) + b

                self.summaries.append(tf.summary.histogram("Value", self.value_current))


        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.value_target, self.value_current))
            self.summaries.append(tf.summary.scalar(('%s_critic_loss' % self.name), self.loss))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


        self.merged_summary = tf.summary.merge(self.summaries)


    def predict(self, state):
        values = self.predict_batch([state])
        return values[0][0]


    def predict_batch(self, batch):
        values = self.session.run(self.value_current, feed_dict = {self.state : batch})
        return values


    def generate_summaries(self, states, steps,  value_target = None):
        feed_dict = {self.state: states}

        if value_target is not None:
            feed_dict[self.value_target] = value_target

        summary_str = self.session.run(self.merged_summary, feed_dict = feed_dict)
        self.summaries_writer.add_summary(summary_str, steps)


    def fit(self, states, value_target, steps):
        if steps % 100 == 0:
            self.generate_summaries(states, steps, value_target)


        _  = self.session.run(self.train_op, feed_dict = {
                                                        self.state: states,
                                                        self.value_target: value_target
                                                      })

    def get_params(self):
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        return params

    def replace(self, from_model):
        t_params = self.get_params()
        f_params = from_model.get_params()
        self.session.run([ tf.assign(t, f) for t, f in zip(t_params, f_params) ])
