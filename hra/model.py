import tensorflow as tf
import numpy as np

class HRAModel(object):
    """Neural Network with the HRA architecture  """
    def __init__(self, size_features, size_actions, size_rewards, name = "default_model", learning_rate = 0.001):
        super(HRAModel, self).__init__()
        self.name = name
        self.size_features = size_features
        self.size_actions  = size_actions
        self.size_rewards  = size_rewards

        self.learning_rate = learning_rate #TODO: Change to expoential decay?
        self.summeries = []
        self.build_network()

    def build_network(self):
        n_h1 = 25
        n_h2 = 25

        self.state = tf.placeholder(tf.float32, [None, self.size_features], name="State")
        self.q_target = tf.placeholder(tf.float32, [None, self.size_rewards, self.size_actions])


        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        collections = [tf.GraphKeys.GLOBAL_VARIABLES, self.name + "_Collection" ]

        with tf.variable_scope(self.name):

            with tf.variable_scope("CommonHiddenLayer1"):
                w1 = tf.get_variable("W1", [self.size_features, n_h1], initializer = w_initializer, collections = collections)
                self.summeries.append(tf.summary.histogram('W1', w1))
                b1 = tf.get_variable("B1", [1, n_h1], initializer = b_initializer, collections = collections)
                self.summeries.append(tf.summary.histogram('B1', b1))
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            # with tf.variable_scope("CommonHiddenLayer2"):
            #     w2 = tf.get_variable("W2", [n_h1, n_h2], initializer = w_initializer, collections = collections)
            #     self.summeries.append(tf.summary.histogram('W2', w2))
            #     b2 = tf.get_variable("B2", [1, n_h2], initializer = b_initializer, collections = collections)
            #     self.summeries.append(tf.summary.histogram('B2', b2))
            #     l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            self.q_reward = []

            for reward in range(self.size_rewards):
                with tf.variable_scope("OutputLayer_Reward_%d" % (reward + 1)):
                    #TODO
                    # w3 = tf.get_variable("W3", [n_h1, self.size_actions], initializer = w_initializer, collections = collections)
                    # self.summeries.append(tf.summary.histogram("W3_R%d" % reward, w3))
                    # b3 = tf.get_variable("B3", [1, self.size_actions], initializer = b_initializer, collections = collections)
                    # self.summeries.append(tf.summary.histogram("B3_R%d" %reward, b3))
                    # l3 = tf.matmul(l1, w3) + b3 #TODO
                    # name = 'Q_R_%d' % reward
                    # self.summeries.append(tf.summary.histogram(name, l3))
                    # self.q_reward.append(l3)
                     w2 = tf.get_variable("W2", [n_h1, self.size_actions], initializer = w_initializer, collections = collections)
                     self.summeries.append(tf.summary.histogram('W2', w2))
                     b2 = tf.get_variable("B2", [1, self.size_actions], initializer = b_initializer, collections = collections)
                     self.summeries.append(tf.summary.histogram('B2', b2))
                     l2 = tf.matmul(l1, w2) + b2
                     name = 'Q_R_%d' % reward
                     self.summeries.append(tf.summary.histogram(name, l2))
                     self.q_reward.append(l2)

            self.q_current = tf.stack(self.q_reward, axis = 1)

            self.q_values = tf.reduce_mean(self.q_current, axis = 1)

            # for reward in range(self.size_rewards):
            #     for action in range(self.size_actions):
            #         self.summeries.append(tf.summary.histogram('R_%d_Q%d' % (reward, action), self.q_reward[reward][action]))


            for action in range(self.size_actions):
                name =  'Mean_Q%d' % action
                self.summeries.append(tf.summary.histogram(name, self.q_values[action]))

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_current))
                self.summeries.append(tf.summary.scalar('loss', self.loss))

            with tf.variable_scope('train'):
                self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.merged_summary = tf.summary.merge(self.summeries)

    def predict(self, state, session):
        _, q_values = self.predict_batch([state], session)
        action = np.argmax(q_values[0])
        return action

    def predict_batch(self, batch, session):
        q_current, q_values, = session.run([self.q_current, self.q_values], feed_dict = {self.state : batch})
        return (q_current, q_values)

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
