import numpy as np
import torch
import time
import copy
import threading
import visdom

from multiprocessing import Queue


class PDX(object):
    """ Predictive Difference Explanations (PDXs) """

    def __init__(self):
        super(PDX, self).__init__()
        self.pdx_box = {}
        self.min_pdx_box = {}
        self.viz = visdom.Visdom()

    def get_pdx(self, q_values, selected_action, target_actions):
        """predicted delta explanations for the selected action in comparision with the target action.

        :param list q_values: decomposed q-values
        :param int selected_action:
        :param list target_actions:

        >>> PDX().get_pdx(np.array([[1, 1, 1, 1], [4, 3, 2, 1]]), 0, [1, 2])
        [[0, 0], [1, 2]]

        >>> PDX().get_pdx(np.array([[1, 0, 2, 1], [3, 1, 2, 0]]), 0, [1, 2])
        [[1, -1]]

        """
        pdx = [[(q_values[r][selected_action] - q_values[r][target]) for target in target_actions]
               for r in range(len(q_values))]
        return pdx


    def get_mse(self, pdx):
        sorted = all(pdx[i] >= pdx[i+1] for i in range(len(pdx)-1))
        if not sorted:
            raise "The pdx should be sorted in decending order!!!!!"
        positive_pdx = [pdx[i] for i in range(len(pdx)) if pdx[i] > 0]
        negative_pdx = [pdx[i] for i in range(len(pdx)) if pdx[i] < 0]

        if len(negative_pdx) == 0:
            return []

        for i in range(len(pdx)):
            if sum(positive_pdx[:i]) > abs(sum(negative_pdx)):
                return positive_pdx[:i]
        return pdx


    def mse_pdx(self, prediction_x, target_x):
        """Mean Square Error between Predicted and Target explanations

        >>> PDX().mse_pdx([[0, 0], [1, 2]],[[0, 0], [1, 2]])
        0.0
        >>> PDX().mse_pdx([[3, -2], [1, 2]],[[0, 0], [1, 2]])
        3.25
        """

        return np.square(np.subtract(prediction_x, target_x)).mean()

    def clear_windows(self):
        for box in self.pdx_box.values():
            self.viz.close(box)

        for box in self.min_pdx_box.values():
            self.viz.close(box)


    def render_all_pdx(self, current_action, action_space, q_values, action_names, reward_types):
        self.clear_windows()
        for target_action in range(action_space):
            if current_action != target_action:
                self.render_pdx(q_values, current_action, target_action, action_names, reward_types)

    def render_pdx(self, q_values, current_action, target_action, action_names, reward_types):
        action_name = action_names[current_action]
        target_action_name = action_names[target_action]
        title = "PDX (%s, %s)" % (action_name, target_action_name)

        pdx = self.get_pdx(q_values, current_action, [target_action])
        pdx = np.array(pdx).squeeze()
        sorted_pdx, reward_names = zip(*sorted(zip(pdx, reward_types), key= lambda x: -x[0]))

        pdx_box_opts = dict(
            title = title,
            stacked = False,
            legend = reward_names
        )

        if (current_action, target_action) not in self.pdx_box:
            self.pdx_box[(current_action, target_action)] = self.viz.bar(X=sorted_pdx, opts=pdx_box_opts)
        else:
            self.viz.bar(X=sorted_pdx, opts=pdx_box_opts, win=self.pdx_box[(current_action, target_action)])

        min_pdx = self.get_mse(sorted_pdx)
        min_pdx = list(min_pdx) + [0] * (len(sorted_pdx) - len(min_pdx))

        pdx_box_opts = dict(
            title = "MSE PDX (%s, %s)" % (action_name, target_action_name),
            stacked = False,
            legend = reward_names
        )

        if (current_action, target_action) not in self.min_pdx_box:
            self.min_pdx_box[(current_action, target_action)] = self.viz.bar(X=min_pdx, opts=pdx_box_opts)
        else:
            self.viz.bar(X=min_pdx, opts=pdx_box_opts, win=self.min_pdx_box[(current_action, target_action)])



if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)
