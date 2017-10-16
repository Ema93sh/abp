import numpy as np
import matplotlib.pyplot as plt
import time

plt.ion()
fig, ax = plt.subplots()

class SingleQBarChart(object):
    """Bar chart for single q values"""
    def __init__(self, action_size, action_names):
        super(SingleQBarChart, self).__init__()
        self.action_size = action_size
        self.action_names = action_names #TODO
        self.bar = None


    def render(self, q_values, title = 'Why did I make the move?'):
        if self.bar is None:
            x_pos = np.arange(self.action_size)
            self.bar = ax.bar(x_pos, q_values, align='center',
                            color='green', ecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(self.action_names)
            ax.set_xlabel('Q Value')
            ax.set_title(title)
        else:
            for i, b in enumerate(self.bar):
                b.set_height(q_values[i])

        plt.show()
        plt.pause(0.001)

        pass
