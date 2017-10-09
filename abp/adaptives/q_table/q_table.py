import numpy as np

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
