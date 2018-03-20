import numpy as np

class EnvMap(object):
    """EnvMap object used to load and render map from file"""

    def __init__(self, map_path):
        super(EnvMap, self).__init__()
        self.map_path = map_path
        self.load_map()


    def get_traveller_locations(self):
        locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_traveller(row, col):
                    locations.append((row, col))
        return locations

    def get_gold_locations(self):
        locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_gold(row, col):
                    locations.append((row, col))
        return locations

    def get_diamond_locations(self):
        locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_diamond(row, col):
                    locations.append((row, col))
        return locations

    def get_hill_locations(self):
        locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_hill(row, col):
                    locations.append((row, col))
        return locations

    def get_mountain_locations(self):
        locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_mountain(row, col):
                    locations.append((row, col))
        return locations

    def get_river_locations(self):
        locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_river(row, col):
                    locations.append((row, col))
        return locations

    def get_home_locations(self):
        locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_home(row, col):
                    locations.append((row, col))
        return locations


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

    def has_gold(self, row, col):
        return "G" == self.grid[row][col]

    def has_diamond(self, row, col):
        return "D" == self.grid[row][col]

    def has_traveller(self, row, col):
        return "T" == self.grid[row][col]

    def has_hill(self, row, col):
        return "H" == self.grid[row][col]

    def has_mountain(self, row, col):
        return "M" == self.grid[row][col]

    def has_river(self, row, col):
        return "R" == self.grid[row][col]

    def has_home(self, row, col):
        return "F" == self.grid[row][col]

    def shape(self):
        return (len(self.grid), len(self.grid[0]))
