import os
from itertools import chain

import numpy as np
from six import StringIO
from gym import spaces, Env

from .env_map import EnvMap

class WolfHuntEnv(Env):
    metadata = {'render.modes': ['ansi'], 'state.modes': ['linear', 'grid', 'channels']}
    LEFT, RIGHT, UP, DOWN, NOOP = [0, 1, 2, 3, 4]
    ACTIONS = [LEFT, RIGHT, UP, DOWN, NOOP]
    action_map = {
        UP: "UP",
        DOWN: "DOWN",
        LEFT: "LEFT",
        RIGHT: "RIGHT",
        NOOP: "NOOP",
    }

    "Wolves hunting prey"

    def __init__(self, map_name = "10x10_default"):
        super(WolfHuntEnv, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.current_dir = os.path.dirname(os.path.realpath(__file__))

        self.reset(map_name)


    def reset(self, map_name = "10x10_default", state_mode = "linear"):
        self.state_mode = state_mode
        self.map = EnvMap(os.path.join(self.current_dir, "maps", map_name + ".map"))

        self.current_predator_positions = self.map.get_predator_locations()
        self.predators = sorted(self.current_predator_positions.keys())

        self.current_prey_positions = self.map.get_prey_locations()
        self.preys = sorted(self.current_prey_positions.keys())
        return self.generate_state()

    def generate_state(self):
        #TODO make sure its ordered!!!!!!!!!!!!
        generate = {
            "linear": self.generate_linear_state,
            "grid": self.generate_grid_state,
            "channels": self.generate_channels_state
        }
        return generate[self.state_mode]()

    def generate_linear_state(self):
        shape = self.map.shape()
        predator_state = np.zeros(shape)
        for p, location in self.current_predator_positions.items():
            if location:
                predator_state[location] = 1

        prey_state = np.zeros(shape)
        for p, location in self.current_prey_positions.items():
            if location:
                prey_state[location] = 1

        return np.concatenate((predator_state.reshape(np.prod(shape)), prey_state.reshape(np.prod(shape))))

    def generate_grid_state(self):
        # (2, 10, 10) grid
        # one for predator locations
        # one for prey location
        shape = self.map.shape()
        predator_state = np.zeros(shape)
        for p, location in self.current_predator_positions.items():
            if location:
                predator_state[location] = 1


        prey_state = np.zeros(shape)
        for p, location in self.current_prey_positions.items():
            if location:
                prey_state[location] = 1

        return np.stack((predator_state, prey_state))

    def generate_channels_state(self):
        # (n, 10, 10) grid
        # channel for each animal

        shape = self.map.shape()
        channels = []


        for p, location in self.current_predator_positions.items():
            predator_state = np.zeros(shape)
            if location:
                predator_state[location] = 1
            channels.append(predator_state)


        for p, location in self.current_prey_positions.items():
            prey_state = np.zeros(shape)
            if location:
                prey_state[location] = 1
            channels.append(prey_state)

        return np.stack(channels)

    def update_location(self, location, action):
        x, y = location
        if action == self.NOOP:
            return location
        elif action == self.LEFT: #LEFT
            y = y - 1
        elif action == self.RIGHT: #RIGHT
            y = y + 1
        elif action == self.UP: #UP
            x = x - 1
        elif action == self.DOWN: #DOWN
            x = x + 1
        else:
            raise "Invalid Action"
        return (x, y)

    def update_prey_pos(self, prey, action):
        current_pos = self.current_prey_positions[prey]
        x, y = self.update_location(current_pos, action)


        if self.map.has_wall(x, y):
            return current_pos


        #should not move to different prey
        for p in self.preys:
            if p != prey and self.current_prey_positions[p] == (x, y):
                return current_pos


        #should not move to predator
        for p in self.predators:
            if self.current_predator_positions[p] == (x, y):
                return current_pos


        self.current_prey_positions[prey] = (x, y)
        return self.current_prey_positions[prey]

    def update_predator_pos(self, predator, action):
        current_pos = self.current_predator_positions[predator]
        x, y = self.update_location(current_pos, action)

        if self.map.has_wall(x, y):
            return current_pos

        for p in self.predators:
            if self.current_predator_positions[p] == (x, y):
                return current_pos

        self.current_predator_positions[predator] = (x, y)

        return self.current_predator_positions[predator]

    def step(self, predator_action, decomposed_reward = False):
        # Action is mapping from predator to action
        info  = {}
        reward = 0


        for p, action in predator_action.items():
            updated_pos = self.update_predator_pos(p, action)
            for prey in self.preys:
                if updated_pos == self.current_prey_positions[prey]:
                    reward += 1
                    self.current_prey_positions[prey] = None


        for prey in self.preys:
            action = np.random.choice(self.ACTIONS)
            if self.current_prey_positions[prey] is not None:
                updated_pos = self.update_prey_pos(prey, action)


        all_prey_killed = all(map(lambda x: x is None, self.current_prey_positions.values()))

        return self.generate_state(), reward, all_prey_killed, info

    def is_prey(self, i, j):
        for prey in self.preys:
            if self.current_prey_positions[prey] == (i, j):
                return prey
        return None

    def is_predator(self, i, j):
        for predator in self.predators:
            if self.current_predator_positions[predator] == (i, j):
                return predator
        return None

    def render_ansi(self):
        output = StringIO()
        row, col = self.map.shape()

        for i in range(row):
            for j in range(col):
                prey =  self.is_prey(i, j)

                predator = self.is_predator(i, j)

                if prey:
                    output.write(" %s " % prey)
                elif predator:
                    output.write(" %s " % predator)
                elif self.map.has_wall(i, j):
                    output.write(" * ")
                else:
                    output.write(" - ")

            output.write("\n")

        return output


    def render(self, mode = 'ansi', close = False):
        if close:
            return None

        return self.render_ansi()
