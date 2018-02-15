import os
import logging
logger = logging.getLogger('root')

import tensorflow as tf
import numpy as np

from abp.utils import clear_summary_path


#TODO
# * Lot of duplicate Code. Only the build network differs?

class HRAModel(object):
    """Neural Network with the HRA architecture  """
    def __init__(self, name, network_config, session, restore = True, learning_rate = 0.001):
        super(HRAModel, self).__init__()
        self.name = name
        self.network_config = network_config
        self.collections = []
        self.reward_types = len(self.network_config.networks)
        self.restore = restore

        # TODO add ability to configure learning rate for network!
        self.learning_rate = learning_rate

        self.summeries = []

        self.session = session

        logger.info("Building network for %s" % self.name)

        self.build_network()

        self.saver = tf.train.Saver()

        self.session.run(tf.global_variables_initializer())

        # TODO
        # * Option to disable summeries

        clear_summary_path(self.network_config.summaries_path + "/" + self.name)

        self.summaries_writer = tf.summary.FileWriter(self.network_config.summaries_path + "/" + self.name)

        logger.info("Created network for %s" % self.name)

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
            self.saver.restore(self.session, self.network_config.network_path + "/" + self.name)

    def build_network(self):
        w_initializer = tf.random_normal_initializer(0.0, 0.1)
        b_initializer = tf.constant_initializer(0.1)

        self.collections = [tf.GraphKeys.GLOBAL_VARIABLES, self.name + "_Collection"]

        q_rewards = []

        #TODO
        # * common input state. Option to have separate for each network
        # * generate shared hidden layer
        self.q_target = tf.placeholder(tf.float32, [len(self.network_config.networks), None] + self.network_config.output_shape)
        self.state = tf.placeholder(tf.float32,
                                     shape = [None] + self.network_config.input_shape,
                                     name = self.name + "_input_state")
        self.loss_ops = []
        self.train_ops = []

        with tf.variable_scope(self.name):
            #TODO Common shared layers!
            shape = self.state.get_shape().as_list()        # a list: [None, 9, 2]
            input_layer_size = np.prod(shape[1:])                       # input_layer_size = prod(9,2) = 18
            input_layer = tf.reshape(self.state, [-1, input_layer_size])

            for network_id, network in enumerate(self.network_config.networks):
                L = []
                with tf.variable_scope(network["name"]):

                    with tf.variable_scope("Hidden_Layer"):
                        # First Hidden Layer
                        first_layer_size = network["layers"][0]

                        w = tf.get_variable("w1", shape = (input_layer_size, first_layer_size),
                                             initializer = w_initializer,
                                             collections = self.collections)

                        b = tf.get_variable("b1", shape = (1, first_layer_size),
                                             initializer = b_initializer,
                                             collections = self.collections)

                        L.append(tf.nn.relu(tf.matmul(input_layer, w) + b))

                        # Generate other Hidden Layers
                        for layer_id in range(1, len(network["layers"])):
                            previous_layer_size = network["layers"][layer_id-1]
                            current_layer_size  = network["layers"][layer_id]

                            w = tf.get_variable("w" + str(layer_id+1),
                                            shape = (previous_layer_size, current_layer_size),
                                            initializer = w_initializer,
                                            collections = self.collections)

                            b = tf.get_variable("b" + str(layer_id+1),
                                            shape = (1, current_layer_size),
                                            initializer = b_initializer,
                                            collections = self.collections)

                            l = tf.nn.relu(tf.matmul(L[layer_id-1], w) + b)
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

                        with tf.variable_scope('loss'):
                            loss = tf.reduce_mean(tf.squared_difference(self.q_target[network_id], q))
                            self.loss_ops.append(loss)
                            self.summeries.append(tf.summary.scalar('%s_loss' % network["name"], loss))

                        with tf.variable_scope('train'):
                            train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
                            self.train_ops.append(train_op)

            self.q_current = tf.stack(q_rewards, axis = 0)

            self.merged_summary = tf.summary.merge(self.summeries)

    def predict(self, state): # TODO multiple state for each network
        q_heads = self.predict_batch([state])
        q_heads = np.stack(q_heads, axis=1)[0]
        q_heads = np.stack(q_heads, axis=0)
        merged = np.sum(q_heads, axis=0)
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


        _  = self.session.run(self.train_ops, feed_dict = {
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
