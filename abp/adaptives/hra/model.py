import os
import logging

import tensorflow as tf
import numpy as np

from abp.utils import clear_summary_path


#TODO
# * Lot of duplicate Code. Only the build network differs?

class HRAModel(object):
    """Neural Network with the HRA architecture  """
    def __init__(self, name, network_config, learning_rate = 0.001):
        super(HRAModel, self).__init__()
        self.name = name
        self.network_config = network_config
        self.collections = []
        self.reward_types = len(self.network_config.networks)
        self.graph = tf.Graph()

        # TODO add ability to configure learning rate for network!
        self.learning_rate = learning_rate

        self.summeries = []

        with self.graph.as_default():
            self.build_network()

            self.session = tf.Session()

            self.saver = tf.train.Saver()

            #TODO:
            # * This session should be independent. Use current model collection instead

            self.session.run(tf.global_variables_initializer())
            # map(lambda v: v.initializer, tf.get_collection(self.name + "_Collection")
            # self.session.run(map(lambda v: v.initializer, tf.get_collection(self.name + "_Collection")))



        # TODO
        # * Option to disable summeries

        clear_summary_path(self.network_config.summaries_path + "/" + self.name)

        self.summaries_writer = tf.summary.FileWriter(self.network_config.summaries_path + "/" + self.name, graph = self.graph)



        print "Created network for...", self.name

        self.restore_network()

    def __del__(self):
        self.session.close()
        self.summaries_writer.close()


    def save_network(self):
        if self.network_config.network_path:
            dirname = os.path.dirname(self.network_config.network_path + "/" + self.name)
            if not tf.gfile.Exists(dirname):
                logging.info("Creating network path directories...")
                tf.gfile.MakeDirs(dirname)
            logging.info("Saving the network at %s" % self.network_config.network_path + "/" + self.name)
            self.saver.save(self.session, self.network_config.network_path + "/" + self.name)


    def restore_network(self):
        if self.network_config.restore_network and self.network_config.network_path:
            dirname = os.path.dirname(self.network_config.network_path + "/" + self.name)
            if not tf.gfile.Exists(dirname):
                logging.error("Can not restore model. Reason: The network path (%s) does not exists" % self.network_config.network_path)
                return
            self.saver.restore(self.session, self.network_config.network_path + "/" + self.name)

    def build_network(self):
        w_initializer = tf.random_normal_initializer(0.0, 0.1)
        b_initializer = tf.constant_initializer(0.1)

        self.collections = [tf.GraphKeys.GLOBAL_VARIABLES, self.name + "_Collection"]

        q_rewards = []

        #TODO
        # * common input state. Option to have separate for each network
        # * generate shared hidden layer
        self.q_target = tf.placeholder(tf.float32, [None, len(self.network_config.networks), self.network_config.output_shape[0]])
        self.state = tf.placeholder(tf.float32,
                                     shape = [None, self.network_config.input_shape[0]],
                                     name = self.name + "_input_state")

        with tf.variable_scope(self.name):
            for i, network in enumerate(self.network_config.networks):
                L = []
                with tf.variable_scope(network["name"]):

                    with tf.variable_scope("Hidden_Layer"):
                        # First Hidden Layer
                        first_layer_size = network["layers"][0]

                        w = tf.get_variable("w1", shape = (self.network_config.input_shape[0], first_layer_size),
                                             initializer = w_initializer,
                                             collections = self.collections)

                        b = tf.get_variable("b1", shape = (1, first_layer_size),
                                             initializer = b_initializer,
                                             collections = self.collections)

                        L.append(tf.nn.relu(tf.matmul(self.state, w) + b))

                        # Generate other Hidden Layers
                        for i in range(1, len(self.network_config.layers)):
                            previous_layer_size = self.network_config.layers[i-1]
                            current_layer_size = self.network_config.layers[i]

                            w = tf.get_variable("w" + str(i+1),
                                            shape = (previous_layer_size, current_layer_size),
                                            initializer = w_initializer,
                                            collections = self.collections)

                            b = tf.get_variable("b" + str(i+1),
                                            shape = (1, current_layer_size),
                                            initializer = b_initializer,
                                            collections = self.collections)

                            l = tf.nn.relu(tf.matmul(L[i-1], w) + b)
                            L.append(l)

                        with tf.variable_scope("Output_Layer"): #TODO the output layer should be the same for all networks?
                            w = tf.get_variable("w_output",
                                        shape = (network["layers"][-1], self.network_config.output_shape[0]),
                                        initializer = w_initializer,
                                        collections = self.collections)

                            b = tf.get_variable("b_output",
                                                shape = (1, self.network_config.output_shape[0]),
                                                initializer = b_initializer,
                                                collections = self.collections)
                            q = tf.matmul(L[-1], w) + b
                            self.summeries.append(tf.summary.histogram("%s_Q" % network["name"], q))
                            q_rewards.append(q)

            self.q_current = tf.stack(q_rewards, axis = 1)

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_current))
                self.summeries.append(tf.summary.scalar('loss', self.loss))

            with tf.variable_scope('train'):
                self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

            self.merged_summary = tf.summary.merge(self.summeries)

    def predict(self, state): # TODO multiple state for each network
        q_heads = self.predict_batch([state])
        q_heads = q_heads[0]
        merged = np.sum(q_heads * (1.0/self.reward_types), axis=0)
        action = np.argmax(merged)
        return action, q_heads

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
