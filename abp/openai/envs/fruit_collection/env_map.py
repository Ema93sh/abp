import numpy as np

class EnvMap(object):
    """EnvMap object used to load and render map from file"""

    def __init__(self, map_path):
        super(EnvMap, self).__init__()
        self.map_path = map_path
        self.load_map()

    def get_possible_fruit_locations(self):
        possible_fruit_location = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_fruit_location(row, col):
                    possible_fruit_location.append( (row, col) )
        return possible_fruit_location

    def load_map(self):
        self.grid = []

        with open(self.map_path) as fp:
            line = fp.readline()
            while line:
                row = list(map(int, line.strip().split(' ')))
                self.grid.append(row)
                line = fp.readline()

    def render(self):
        for row in self.grid:
            print(row)

    def agent_location(self):
         for row in range(len(self.grid)):
             for col in range(len(self.grid[row])):
                 if self.is_agent_position(row, col):
                     return (row, col)
         return None

    def has_wall(self, row, col):
        return self.grid[row][col] == 1

    def has_fruit_location(self, row, col):
        return self.grid[row][col] == 2

    def is_agent_position(self, row, col):
        return self.grid[row][col] == 3

    def shape(self):
        return (len(self.grid), len(self.grid[0]))
