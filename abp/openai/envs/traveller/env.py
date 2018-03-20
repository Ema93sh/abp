import os
from six import StringIO
from copy import deepcopy

import numpy as np
import gym

from gym import spaces, Env
from .env_map import EnvMap

class TravellerEnv(Env):
    """
    A simple gridworld problem. The objective of the agent to collect rewards and
    reach the house.

    ACTIONS:
    0 - LEFT
    1 - RIGHT
    2 - UP
    3 - DOWN

    Terrains Costs:
    --------------

    Mountain - 4
    Hill - 2
    River - 2

    """
    metadata = {'render.modes': ['ansi'], 'state.modes': ['linear', 'grid', 'channels']} #TODO render to RGB
    LEFT, RIGHT, UP, DOWN, NOOP = [0, 1, 2, 3, 4]

    def __init__(self):
        super(TravellerEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.days = {
                     'mountain': 4,
                     'hill': 3,
                     'river': 2
                    }

        self.treasure = {
                         'gold': 2,
                         'diamond': 3
                         }

        self.current_dir = os.path.dirname(os.path.realpath(__file__))

        self.reset()

    def reset(self, map_name = "5x2_default", state_mode = "linear", days_remaining = 8):
        self.map = EnvMap(os.path.join(self.current_dir, "maps", map_name + ".map"))

        self.gold_locations = self.map.get_gold_locations()
        self.diamond_locations = self.map.get_diamond_locations()

        self.hill_locations = self.map.get_hill_locations()
        self.mountain_locations = self.map.get_mountain_locations()
        self.river_locations = self.map.get_river_locations()

        self.traveller_location = self.map.get_traveller_locations()[0]
        self.house_locations = self.map.get_home_locations()

        self.current_gold_locations = deepcopy(self.gold_locations)
        self.current_diamond_locations = deepcopy(self.diamond_locations)

        self.days_remaining = days_remaining
        self.state_mode = state_mode

        return self.generate_state()

    def one_shot_encode(self, locations):
        shape = self.map.shape()
        grid = np.zeros(shape)
        for loc in locations:
            grid[loc] = 1
        return grid.reshape(np.prod(shape)).tolist()


    def generate_state(self):
        #TODO days remaining is not part of the state
        generate = {
            "linear": self.generate_linear_state,
            "grid": self.generate_grid_state,
            "channels": self.generate_channels_state
        }
        return generate[self.state_mode]()

    def generate_linear_state(self):
        traveller = self.one_shot_encode([self.traveller_location])
        home = self.one_shot_encode(self.house_locations)
        mountain = self.one_shot_encode(self.mountain_locations)
        river = self.one_shot_encode(self.river_locations)
        hill = self.one_shot_encode(self.hill_locations)
        gold = self.one_shot_encode(self.current_gold_locations)
        daimond = self.one_shot_encode(self.current_diamond_locations)

        return traveller + home + mountain + river + hill + gold + daimond + [self.days_remaining]

        # shape = self.map.shape()
        #
        # terrain_state = np.zeros(shape)
        # for location in (self.hill_locations + self.mountain_locations + self.river_locations):
        #     terrain_state[location] = 1
        #
        # treasure_state = np.zeros(shape)
        # for location in (self.current_gold_locations + self.current_diamond_locations):
        #     treasure_state[location] = 1
        #
        # home_state = np.zeros(shape)
        # for location in self.house_locations:
        #     home_state[location] = 1
        #
        # traveller_state = np.zeros(shape)
        # traveller_state[self.traveller_location] = 1
        #
        # days_remaining = np.zeros(shape)
        # days_remaining += self.days_remaining
        #
        # reshape = list(map(lambda x: x.reshape(np.prod(shape)), [terrain_state, treasure_state, home_state, traveller_state, days_remaining]))
        # return np.concatenate(reshape)


    def generate_grid_state(self):
        shape = self.map.shape()

        terrain_state = np.zeros(shape)
        terrain_state[self.hill_locations] = 1
        terrain_state[self.mountain_locations] = 1
        terrain_state[self.river_locations] = 1

        treasure_state = np.zeros(shape)
        treasure_state[self.current_gold_locations] = 1
        treasure_state[self.current_diamond_locations] = 1

        home_state = np.zeros(shape)
        home_state[self.house_locations] = 1

        traveller_state = np.zeros(shape)
        traveller_state[self.traveller_location] = 1

        return np.stack((terrain_state, treasure_state, home_state, traveller_state))

    def generate_channels_state(self):
        shape = self.shape
        channels = []

        gold_state = np.zeros(shape)
        gold_state[self.current_gold_locations] = 1

        diamond_state = np.zeros(shape)
        diamond_state[self.current_diamond_locations] = 1

        hill_state = np.zeros(shape)
        hill_state[self.hill_locations] = 1

        mountain_state = np.zeros(shape)
        mountain_state[self.mountain_locations] = 1

        river_state = np.zeros(shape)
        river_state[self.river_locations] = 1

        home_state = np.zeros(shape)
        home_state[self.home_locations] = 1

        traveller_state = np.zeros(shape)
        traveller_state[self.traveller_location] = 1

        return np.stack((gold_state, diamond_state, hill_state, mountain_state, river_state, home_state, traveller_state))

    def next_location(self, action):
        x, y = self.traveller_location

        if action == self.NOOP:
            return self.traveller_location
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


    def step(self, action, decompose_level = 0):
        # level 0 : no decomposition
        # level 1 : home, treasure, terrain
        # level 2 : home, gold, diamond, hill, mountain, river

        info = {}
        done = False
        terrain_reward  = [0, 0, 0]
        treasure_reward = [0, 0]
        home_reward = 0
        death_reward = 0
        days_reward = 0

        updated_location =  self.next_location(action)

        if self.map.has_wall(*updated_location):
            days_reward -= 1
        else:
            self.traveller_location = updated_location

            if self.traveller_location in self.hill_locations:
                terrain_reward[0] -= self.days['hill']
                days_reward -= self.days['hill']

            elif self.traveller_location in self.mountain_locations:
                terrain_reward[1] -= self.days['mountain']
                days_reward -= self.days['mountain']

            elif self.traveller_location in self.river_locations:
                terrain_reward[2] -= self.days['river']
                days_reward -= self.days['river']

            elif self.traveller_location in self.current_gold_locations:
                treasure_reward[0] += self.treasure['gold']
                idx = self.current_gold_locations.index(self.traveller_location)
                del self.current_gold_locations[idx]
                days_reward -= 1
            elif self.traveller_location in self.current_diamond_locations:
                treasure_reward[1] += self.treasure['diamond']
                idx = self.current_diamond_locations.index(self.traveller_location)
                del self.current_diamond_locations[idx]
                days_reward -= 1
            elif self.traveller_location in self.house_locations:
                home_reward += 10
                days_reward -= 1
            else:
                days_reward -= 1

        self.days_remaining += days_reward

        done = self.days_remaining <= 0 or self.traveller_location in self.house_locations

        if done and self.traveller_location not in self.house_locations:
            home_reward -= 10


        info["days_remaining"] = self.days_remaining

        if decompose_level == 0:
            reward = sum(terrain_reward) + sum(treasure_reward) + home_reward + death_reward
        if decompose_level == 1:
            reward = [sum(terrain_reward), sum(treasure_reward), home_reward, death_reward]
        if decompose_level == 2:
            reward = terrain_reward + treasure_reward + [home_reward] + [death_reward]

        return self.generate_state(), reward, done, info

    def render_ansi(self):
        shape = self.map.shape()

        outfile = StringIO()
        for i in range(shape[0]):
            for j in range(shape[1]):
                location = (i, j)
                if self.map.has_wall(i, j) == 1:
                    outfile.write(" * ")
                elif location == self.traveller_location:
                    outfile.write(" T ")
                elif location in self.current_gold_locations:
                    outfile.write(" G ")
                elif location in self.current_diamond_locations:
                    outfile.write(" D ")
                elif location in self.hill_locations:
                    outfile.write(" H ")
                elif location in self.mountain_locations:
                    outfile.write(" M ")
                elif location in self.river_locations:
                    outfile.write(" R ")
                elif location in self.house_locations:
                    outfile.write(" F ")
                else:
                    outfile.write(" - ")
            outfile.write('\n')

        return outfile

    def render(self, mode = 'ansi', close = False):
        if close:
            return None

        return self.render_ansi()
