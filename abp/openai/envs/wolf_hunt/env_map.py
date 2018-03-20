import numpy as np

class EnvMap(object):
    """EnvMap object used to load and render map from file"""

    def __init__(self, map_path):
        super(EnvMap, self).__init__()
        self.map_path = map_path
        self.load_map()


    def get_predator_locations(self):
        predator_locations = {}
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_predator(row, col):
                    predator_locations[self.grid[row][col]] = (row, col)
        return predator_locations


    def get_prey_locations(self):
        prey_locations = {}
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_prey(row, col):
                    prey_locations[self.grid[row][col]] = (row, col)
        return prey_locations


    def load_map(self):
        self.grid = []

        with open(self.map_path) as fp:
            line = fp.readline()
            while line:
                row = list(line.strip().split(' '))
                self.grid.append(row)
                line = fp.readline()


    def render(self):
        for row in self.grid:
            print(row)


    def has_wall(self, row, col):
        return self.grid[row][col] == "1"


    def has_predator(self, row, col):
        return "W" in self.grid[row][col]


    def has_prey(self, row, col):
        return "R" in self.grid[row][col]


    def shape(self):
        return (len(self.grid), len(self.grid[0]))
