import sys
import numpy as np
from six import StringIO

import gym
from gym import spaces

class TravellerEnv(gym.Env):
    """
    A simple gridworld problem. The objective of the agent to collect rewards and
    reach the house.

    ACTIONS:
    0 - LEFT
    1 - RIGHT
    2 - UP
    3 - DOWN

    MAP (5 x 2 Grid):
    ----

    Index
    ---------
    | 0 | 1 |
    | 2 | 3 |
    | 4 | 5 |
    | 6 | 7 |
    | 8 | 9 |
    ---------

    Locations
    ---------
    | T | G |
    | G | H |
    | M | G |
    | 6 | R |
    | D | 9 |
    ---------

    Terrains Costs:
    --------------

    Mountain - 4
    Hill - 2
    River - 2

    """
    metadata = {'render.modes': ['human', 'ansi']} #TODO render to RGB
    HOME_TYPE, TREASURE_TYPE, TERRAIN_TYPE = [0, 1, 2]

    def __init__(self):
        super(TravellerEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.shape = (5, 2)
        self.viewer  = None
        self.days = {'mountain': 4,
                     'hill': 3,
                     'river': 2}
        self.treasure = {
                         'gold': 2,
                         'diamond': 3
                         }
        self.gold_locations = np.array([1, 2, 5])
        self.diamond_locations = np.array([8])
        self._seed = 0
        self._reset()

    def _reset(self):
        self.traveller_location = 0
        self.house_location = 9
        self.hill_locations = np.array([3])
        self.mountain_locations = np.array([4])
        self.river_locations = np.array([7])
        self.current_gold_locations = self.gold_locations
        self.current_diamond_locations = self.diamond_locations
        self.days_remaining = 8
        return self.generate_state()

    def one_shot_encode(self, locations):
        grid = [0] * 10
        for loc in locations:
            grid[loc] = 1
        return grid


    def generate_state(self):
        traveller = self.one_shot_encode([self.traveller_location])
        home = self.one_shot_encode([self.house_location])
        mountain = self.one_shot_encode(self.mountain_locations)
        river = self.one_shot_encode(self.river_locations)
        hill = self.one_shot_encode(self.hill_locations)
        gold = self.one_shot_encode(self.current_gold_locations)
        daimond = self.one_shot_encode(self.current_diamond_locations)

        return traveller + home + mountain + river + hill + gold + daimond + [self.days_remaining]

    def next_location(self, action):
        if action == 0: #LEFT
            next_location = self.traveller_location - 1
        elif action == 1: #RIGHT
            next_location = self.traveller_location + 1
        elif action == 2: #UP
            next_location = self.traveller_location - self.shape[1]
        elif action == 3: #DOWN
            next_location = self.traveller_location + self.shape[1]
        else:
            raise "Invalid Action"
        return next_location

    def is_valid_location(self, next_location):
        if next_location < 0: #TOP Border
            return False

        if next_location >= 10: #Bottom
            return False

        if self.traveller_location % self.shape[1] == 0 and next_location + 1 == self.traveller_location: #LEFT Border
            return False

        if (self.traveller_location + 1) % self.shape[1] == 0 and next_location - 1 == self.traveller_location: #RIGHT Border
            return False

        return True


    def _step(self, action):
        done = False
        reward = 0
        decomposed_reward = {
          self.HOME_TYPE : 0,
          self.TREASURE_TYPE: 0,
          self.TERRAIN_TYPE: 0
        }
        info = {}

        updated_location =  self.next_location(action)

        if self.is_valid_location(updated_location):
            self.traveller_location = updated_location

            if updated_location in self.mountain_locations:
                decomposed_reward[self.TERRAIN_TYPE] -= self.days['mountain']
                self.days_remaining -= self.days['mountain']
            elif updated_location in self.river_locations:
                decomposed_reward[self.TERRAIN_TYPE] -= self.days['river']
                self.days_remaining -= self.days['river']
            elif updated_location in self.hill_locations:
                decomposed_reward[self.TERRAIN_TYPE] -= self.days['hill']
                self.days_remaining -= self.days['hill']
            elif updated_location in self.current_gold_locations:
                decomposed_reward[self.TREASURE_TYPE] += self.treasure['gold']
                self.days_remaining -= 1
                index = np.argwhere(self.current_gold_locations == updated_location)
                self.current_gold_locations = np.delete(self.current_gold_locations, index)
            elif updated_location in self.current_diamond_locations:
                decomposed_reward[self.TREASURE_TYPE] += self.treasure['diamond']
                index = np.argwhere(self.current_diamond_locations == updated_location)
                self.current_diamond_locations = np.delete(self.current_diamond_locations, index)
                self.days_remaining -= 1
            elif updated_location == self.house_location:
                decomposed_reward[self.HOME_TYPE] += 10
                self.days_remaining -= 1
            else:
                reward = 0
                self.days_remaining -= 1


            self.traveller_location = updated_location
        else:
            reward -= 1
            self.days_remaining -= 1


        done = self.days_remaining <= 0 or self.traveller_location == self.house_location

        if self.days_remaining <= 0 and self.traveller_location != self.house_location:
            reward -= 10
            decomposed_reward[self.HOME_TYPE] -= 10


        info["days_remaining"] = self.days_remaining
        info["decomposed_reward"] = decomposed_reward

        return self.generate_state(), sum(decomposed_reward.values()), done, info

    def render_human(self):
        from gym.envs.classic_control import rendering

        screen_width = 600
        screen_height = 600
        world_width = 200
        scale = screen_width/world_width
        grid_size = 10
        cell_size = 50
        origin_x = screen_width / 2
        origin_y = screen_height /2


        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Draw grid
            l = origin_x  - grid_size / 2
            r = l + grid_size
            t = origin_y + grid_size / 2
            b = t - grid_size
            grid_background = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            grid_background.set_color(239/255.0,239/255.0,239/255.0)

            self.viewer.add_geom(grid_background)


            for x in range(2):
                for y in range(5):
                    l = origin_x  - (cell_size * 5) + (x * cell_size)
                    r = l + cell_size
                    t = origin_y - (cell_size * 5) + (y * cell_size)
                    b = t - cell_size
                    cell = rendering.PolyLine([(l,b), (l,t), (r,t), (r,b)], True)
                    self.viewer.add_geom(cell)

        # Draw Traveller
        x, y = self.traveller_location / 2, (self.traveller_location % 5 + 1)
        l = origin_x  - (cell_size * 5) + (x * cell_size)
        r = l + cell_size
        t = origin_y - (cell_size * 5) + (y * cell_size)
        b = t - cell_size
        x = origin_x - (cell_size * 5) + (x * cell_size) + cell_size / 2
        y = origin_y - (cell_size * 5) + (y * cell_size) + cell_size / 2

        self.rendered_agent = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        self.rendered_agent.set_color(1, 0, 0)
        self.viewer.add_onetime(self.rendered_agent)

        return self.viewer.render(return_rgb_array = False)

    def render_ansi(self):
        outfile = StringIO()

        for i in range(10):
            if self.traveller_location == i:
                outfile.write(" T ")
            elif i == self.house_location:
                outfile.write(" E ")
            elif i in self.mountain_locations:
                outfile.write(" M ")
            elif i in self.hill_locations:
                outfile.write(" H ")
            elif i in self.river_locations:
                outfile.write(" R ")
            elif i in self.current_gold_locations:
                outfile.write(" G ")
            elif i in self.current_diamond_locations:
                outfile.write(" D ")
            else:
                outfile.write(" - ")

            if (i + 1) % 2 == 0:
                outfile.write("\n")

        outfile.write('\n')

        return outfile


    def _render(self, mode = 'human', close = False):
        if close:
            return None

        if mode == 'human':
            return self.render_human()
        else:
            return self.render_ansi()
