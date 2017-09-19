import tensorflow as tf
import numpy as np

class DQNModel(object):
    """Neural Network for the DQN algorithm """
    def __init__(self, size_features, size_actions, name = "default_model", learning_rate = 0.001, trainable = True):
        super(DQNModel, self).__init__()
        self.name = name
        self.size_features = size_features
        self.size_actions = size_actions
        self.trainable = trainable

        self.learning_rate = learning_rate #TODO: Change to expoential decay?
        self.summaries = []
        self.build_network()

    def build_network(self):
        self.state = tf.placeholder(tf.float32, [None, self.size_features], name= self.name + "_state")
        self.q_target = tf.placeholder(tf.float32, [None, self.size_actions], name= self.name + "_qvalues")

        n_h1 = 25
        n_h2 = 25
        # TODO Random initialization
        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        collections = [tf.GraphKeys.GLOBAL_VARIABLES, self.name + "_Collection"]


        with tf.variable_scope(self.name):

            with tf.variable_scope("HiddenLayer1"):
                w1 = tf.get_variable("W1", [self.size_features, n_h1], initializer = w_initializer, collections = collections)
                self.summaries.append(tf.summary.histogram('W1', w1))
                b1 = tf.get_variable("B1", [1, n_h1], initializer = b_initializer, collections = collections)
                self.summaries.append(tf.summary.histogram('B1', b1))
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.variable_scope("HiddenLayer2"):
                w2 = tf.get_variable("W2", [n_h1, n_h2], initializer = w_initializer, collections = collections)
                self.summaries.append(tf.summary.histogram('W2', w2))
                b2 = tf.get_variable("B2", [1, n_h2], initializer = b_initializer, collections = collections)
                self.summaries.append(tf.summary.histogram('B2', b2))
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope("OutputLayer"):
                w3 = tf.get_variable("W3", [n_h2, self.size_actions], initializer = w_initializer, collections = collections)
                self.summaries.append(tf.summary.histogram('W3', w3))
                b3 = tf.get_variable("B3", [1, self.size_actions], initializer = b_initializer, collections = collections)
                self.summaries.append(tf.summary.histogram('B3', b3))
                self.q_current = tf.matmul(l2, w3) + b3
                self.summaries.append(tf.summary.histogram("Q", self.q_current))
            #
            # for action in range(self.size_actions):
            #     self.summaries.append(tf.summary.histogram('Q%s' % action, self.q_current[action]))


            if self.trainable:
                with tf.variable_scope('loss'):
                    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_current))
                    self.summaries.append(tf.summary.scalar('loss', self.loss))

                with tf.variable_scope('train'):
                    self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


            self.merged_summary = tf.summary.merge(self.summaries)

    def predict(self, state, session):
        q_values = self.predict_batch([state], session)
        action = np.argmax(q_values[0])
        return action

    def predict_batch(self, batch, session):
        q_values, = session.run([self.q_current], feed_dict = {self.state : batch})
        return q_values

    def generate_summaries(self, session, states, writer, steps,  q_target = None):
        feed_dict = {self.state: states}

        if q_target is not None:
            feed_dict[self.q_target] = q_target

        summary_str = session.run(self.merged_summary, feed_dict = feed_dict)
        writer.add_summary(summary_str, steps)


    def fit(self, states, q_target, session, writer, steps):
        if steps % 100 == 0:
            self.generate_summaries(session, states, writer, steps, q_target)


        _  = session.run(self.train_op, feed_dict = {
                                                        self.state: states,
                                                        self.q_target: q_target
                                                      })
