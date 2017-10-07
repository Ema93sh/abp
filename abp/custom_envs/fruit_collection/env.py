import sys
import numpy as np
from six import StringIO

import gym
from gym import spaces

class FruitCollectionEnv(gym.Env):
    """A simple gridworld problem. The objective of the agent to collect all
    the randomly placed fruit in the grid.

    ACTIONS:
    0 - LEFT
    1 - RIGHT
    2 - UP
    3 - DOWN

    """
    metadata = {'render.modes': ['human', 'ansi']} #TODO render to RGB

    def __init__(self):
        super(FruitCollectionEnv, self).__init__()
        self.action_space = spaces.Discrete(4)


        self.number_of_fruits = 5
        self.possible_fruit_locations = [0, 9, 89, 99, 49, 54, 59, 4, 94, 64]
        self.current_fruit_locations = []
        self.agent_location = None
        self.collected_fruit = 0

        self.shape = (10, 10)
        self.current_step = 0
        self._reset()


    def _reset(self):
        self.current_step = 0
        self.collected_fruit = 0

        self.grid = np.ravel(np.zeros(self.shape))

        self.agent_location = np.random.choice(np.setdiff1d(range(100), self.possible_fruit_locations))

        self.grid[self.agent_location] = 1

        self.current_fruit_locations = np.random.choice(self.possible_fruit_locations, self.number_of_fruits, replace = False)

        return self.generate_state()


    def generate_state(self):
        state_fruit = np.array([0] * 10)
        fruit_idx = np.where(np.isin(self.possible_fruit_locations, self.current_fruit_locations))
        state_fruit[fruit_idx] = 1
        return np.concatenate((self.grid, state_fruit))

    def next_location(self, action):
        if action == 0: #LEFT
            next_location = self.agent_location - 1
        elif action == 1: #RIGHT
            next_location = self.agent_location + 1
        elif action == 2: #UP
            next_location = self.agent_location - self.shape[0]
        elif action == 3: #DOWN
            next_location = self.agent_location + self.shape[0]
        else:
            raise "Invalid Action"
        return next_location

    def is_valid_next_location(self, next_location):
        if next_location < 0: #TOP Border
            return False

        if next_location >= 100: #Bottom
            return False

        if self.agent_location % self.shape[0] == 0 and next_location + 1 == self.agent_location: #LEFT Border
            return False

        if (self.agent_location + 1) % self.shape[0] == 0 and next_location - 1 == self.agent_location: #RIGHT Border
            return False

        return True


    def _step(self, action):
        self.current_step += 1
        done = False
        reward = 0
        info = {"current_fruit_locations": self.current_fruit_locations,
                "agent_location": self.agent_location,
                "collected_fruit": None,
                "possible_fruit_locations": self.possible_fruit_locations}
        updated_agent_location =  self.next_location(action)

        if self.is_valid_next_location(updated_agent_location):
            self.grid[self.agent_location] = 0

            if updated_agent_location in self.current_fruit_locations:
                reward = 1
                info["collected_fruit"] = updated_agent_location
                self.collected_fruit += 1
                idx = np.where(self.current_fruit_locations == updated_agent_location)
                self.current_fruit_locations = np.delete(self.current_fruit_locations, idx)

            self.grid[updated_agent_location] = 1
            self.agent_location = updated_agent_location

        done = (self.collected_fruit == self.number_of_fruits or self.current_step >= 300)

        return self.generate_state(), reward, done, info

    def _render(self, mode = 'human', close = False):
        if close:
            return None

        reshaped_board = np.reshape(self.grid, self.shape)

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for i in range(len(self.grid)):
            if self.grid[i] == 1:
                outfile.write(" X ")
            elif i in self.current_fruit_locations:
                outfile.write(" F ")
            else:
                outfile.write(" - ")

            if (i + 1) % 10 == 0:
                outfile.write("\n")

        outfile.write('\n')

        return outfile
