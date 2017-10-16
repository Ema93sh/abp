import numpy as np
import operator
from abp.adaptives.common.q_table import QTable

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

    def save(self, file_path):
        for i, q_table in enumerate(self.q_tables):
            q_table.save(file_path + "_" + str(i))

    def load(self, file_path):
        for i in range(self.reward_size):
            self.q_tables[i].load(file_path + "_" + str(i))
