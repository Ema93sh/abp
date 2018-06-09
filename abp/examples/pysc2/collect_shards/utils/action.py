from functools import partial


import numpy as np
from pysc2.lib import actions, features


FUNCTIONS = actions.FUNCTIONS
_PLAYER_SELF = features.PlayerRelative.SELF


class ActionWrapper(object):
    """Custom ActionWrapper for PySc2 Environment"""

    def __init__(self, obs, grid_size=10):
        super(ActionWrapper, self).__init__()
        self.obs = obs
        self.grid_size = grid_size

    def are_units_selected(self, units_pos):
        "True if all the units are selected"
        selected_units = self.obs[-1].observation.feature_screen.selected
        return all(map(lambda pos: selected_units[pos[0], pos[1]] == 1, units_pos))

    def is_valid(self, x, y):
        if x < 0 or x >= self.grid_size:
            return False
        if y < 0 or y >= self.grid_size:
            return False
        return True

    def move_up(self, marines):
        """ Move selected unit up """
        marine = marines[0]
        x, y = marine.x, marine.y
        y = y - 1
        if not self.is_valid(x, y):
            return None

        return actions.FunctionCall(FUNCTIONS.Move_screen.id, [[0], [x, y]])

    def move_down(self, marines):
        """ Move selected unit down """
        marine = marines[0]
        x, y = marine.x, marine.y
        y = y + 1

        if not self.is_valid(x, y):
            return None

        return actions.FunctionCall(FUNCTIONS.Move_screen.id, [[0], [x, y]])

    def move_left(self, marines):
        """ Move selected unit down """
        marine = marines[0]
        x, y = marine.x, marine.y
        x = x - 1

        if not self.is_valid(x, y):
            return None

        return actions.FunctionCall(FUNCTIONS.Move_screen.id, [[0], [x, y]])

    def move_right(self, marines):
        """ Move selected unit down """
        marine = marines[0]
        x, y = marine.x, marine.y
        x = x + 1

        if not self.is_valid(x, y):
            return None

        return actions.FunctionCall(FUNCTIONS.Move_screen.id, [[0], [x, y]])

    def get_selected_units(self):
        """ returns the indices of the selected units """
        selected_units = self.obs[-1].observation.feature_screen.selected
        idx = np.where(selected_units == 1)
        return np.transpose(idx)

    def select_army(self):
        """ Select the army """
        return actions.FunctionCall(FUNCTIONS.select_army.id, [[0]])

    def select_marine(self, marine):
        x, y = marine.x, marine.y
        return actions.FunctionCall(FUNCTIONS.select_point.id, [[0], [x, y]])

    def select(self, actions):
        marines = [unit for unit in self.obs[-1].observation.feature_units
                   if unit.alliance == _PLAYER_SELF]

        selected_units = [marine for marine in marines if marine.is_selected]

        action_map = {
            "Up": partial(self.move_up, selected_units),
            "Left": partial(self.move_left, selected_units),
            "Right": partial(self.move_right, selected_units),
            "Down": partial(self.move_down, selected_units),
            "SelectMarine1": partial(self.select_marine, marines[0]),  # TODO
            "SelectMarine2": partial(self.select_marine, marines[1]),  # TODO
        }

        actions = map(lambda action: action_map[action](), actions)
        return filter(lambda x: x is not None, actions)
