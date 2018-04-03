import sys
import numpy as np
from six import StringIO

import gym
from gym import spaces
from .env_map import EnvMap
import os


class FruitCollectionEnv(gym.Env):
    """A simple gridworld problem. The objective of the agent to collect all
    the randomly placed fruit in the grid.

    ACTIONS:
    0 - LEFT
    1 - RIGHT
    2 - UP
    3 - DOWN

    """
    metadata = {'render.modes': ['ansi'], 'state.modes': ['linear', 'grid', 'channels']} #TODO render to RGB
    LEFT, RIGHT, UP, DOWN, NOOP = [0, 1, 2, 3, 4]

    def __init__(self, map_name = "10x10_default"):
        super(FruitCollectionEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.max_step = 300
        self.num_envs = 10
        self.reset(map_name)

    def reset(self, map_name = "10x10_default", number_of_fruits = 5, state_mode = "linear"):
        self.number_of_fruits = number_of_fruits
        self.state_mode = state_mode
        self.current_step = 0

        self.map = EnvMap(os.path.join(self.current_dir, "maps", map_name + ".map"))

        self.possible_fruit_locations = self.map.get_possible_fruit_locations()
        self.agent_location           = self.map.agent_location()

        selected_index = np.random.choice(len(self.possible_fruit_locations), self.number_of_fruits, replace = False)
        self.current_fruit_locations = []

        for i in selected_index:
            self.current_fruit_locations.append(self.possible_fruit_locations[i])

        state = self.generate_state()
        self.observation_space = spaces.Box(low= 0, high = 1, shape = state.shape)
        return state

    def generate_state(self):
        generate = {
            "linear": self.generate_linear_state,
            "grid": self.generate_grid_state,
            "channels": self.generate_channels_state
        }
        return generate[self.state_mode]()

    def generate_linear_state(self):
        shape = self.map.shape()
        agent_state = np.zeros(shape)
        agent_state[self.agent_location] = 1

        fruit_state = np.zeros(len(self.possible_fruit_locations))

        for i in range(len(self.possible_fruit_locations)):
            location = self.possible_fruit_locations[i]
            if location in self.current_fruit_locations:
                fruit_state[i] = 1

        return np.concatenate((agent_state.reshape(np.prod(shape)), fruit_state))

    def generate_grid_state(self):
        # (2, 10, 10) grid
        # one for fruit locations
        # one for agent location
        shape = self.map.shape()
        agent_state = np.zeros(shape)
        agent_state[self.agent_location] = 1

        fruit_state = np.zeros(shape)

        for location in self.possible_fruit_locations:
            if location in self.current_fruit_locations:
                fruit_state[location] = 1

        return np.stack((agent_state, fruit_state))

    def generate_channels_state(self):
        # (n, 10, 10) grid
        # on channel for each fruit

        shape = self.map.shape()
        agent_state = np.zeros(shape)
        agent_state[self.agent_location] = 1

        channels = [agent_state]

        for location in self.possible_fruit_locations:
            fruit_state = np.zeros(shape)

            if location in self.current_fruit_locations:
                fruit_state[location] = 1

            channels.append(fruit_state)

        return np.stack(channels)


    def update_agent_location(self, action):
        x, y = self.agent_location

        if action == self.LEFT: #LEFT
            y = y - 1
        elif action == self.RIGHT: #RIGHT
            y = y + 1
        elif action == self.UP: #UP
            x = x - 1
        elif action == self.DOWN: #DOWN
            x = x + 1
        elif action == self.NOOP:
            return self.agent_location
        else:
            print(action)
            raise "Invalid Action"

        if self.map.has_wall(x, y):
            return self.agent_location
        else:
            self.agent_location = (x, y)
            return self.agent_location


    def step(self, action, decompose_reward = False): #TODO clear this mess!
        done = False
        self.current_step += 1

        reward = [0] * len(self.possible_fruit_locations)

        info = {}


        self.update_agent_location(action)

        if self.agent_location in self.current_fruit_locations:
            idx = self.current_fruit_locations.index(self.agent_location)
            r_idx = self.possible_fruit_locations.index(self.agent_location)
            reward[r_idx] = 1
            del self.current_fruit_locations[idx]

        done = len(self.current_fruit_locations) == 0 or self.current_step == self.max_step

        reward = reward if decompose_reward else sum(reward)

        return self.generate_state(), reward, done, info

    def render_ansi(self):
        shape = self.map.shape()

        outfile = StringIO()
        for i in range(shape[0]):
            for j in range(shape[1]):
                location = (i, j)
                if self.map.has_wall(i, j) == 1:
                    outfile.write(" * ")
                elif location == self.agent_location:
                    outfile.write(" X ")
                elif location in self.current_fruit_locations:
                    outfile.write(" F ")
                else:
                    outfile.write(" - ")
            outfile.write('\n')

        return outfile


    def render(self, mode = 'ansi', close = False):
        if close:
            return None

        return self.render_ansi()
