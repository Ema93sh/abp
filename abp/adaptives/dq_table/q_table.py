import numpy as np
import operator

class QTable(object):
    "Class for storing Q values"
    def __init__(self, init, action_size):
        self.q_table = {}
        self.init_value = init
        self.action_size = action_size
        pass

    def get(self, state, action):
        state = str(state)
        if state in self.q_table and action in self.q_table[state]:
            return self.q_table[state][action]
        else:
            return self.init_value

    def get_for(self, state):
        state = str(state)
        if state in self.q_table:
            return self.q_table[state].values()
        else:
            return [self.init_value] * self.action_size

    def qmax(self, state):
        """ Returns the action with max Q value """
        state = str(state)

        if state not in self.q_table:
            return np.random.choice(self.action_size), self.init_value

        return max(self.q_table[state].iteritems(), key=operator.itemgetter(1))


    def put(self, state, action, value):
        state = str(state)
        if state not in self.q_table:
            self.q_table[state] = {}
            for action in range(self.action_size):
                self.q_table[state][action] = self.init_value

        self.q_table[state][action] = value

        return self.q_table[state][action]

    def contains(self, state):
        if str(state) in self.q_table:
            return True
        else:
            return False


    def clear(self):
        self.q_table = {}

    def load(self, filename):
        #TODO
        pass

    def save(self, filename):
        #TODO
        pass

class AggregateQTable(object):
    "Class for managing decomposed Q table"

    def __init__(self, reward_size, action_size, init_value):
        self.reward_size = reward_size
        self.q_tables = [QTable(init_value, action_size) for _ in range(reward_size)]
        pass

    def put(self, reward_type, state, action, value):
        return self.q_tables[reward_type].put(state, action, value)

    def get(self, reward_type, state, action):
        return self.q_tables[reward_type].get(state, action)

    def contains(self, state):
        return self.q_tables[0].contains(state)

    def qmax_merged(self, state):
        q_values = []

        for reward_type in range(self.reward_size):
            r_q_value = self.q_tables[reward_type].get_for(state)
            q_values.append(r_q_value)

        merged_q_values = np.array(q_values).sum(axis = 0)
        action, max_merged_q = max(np.ndenumerate(merged_q_values), key=operator.itemgetter(1))

        return action[0], max_merged_q
