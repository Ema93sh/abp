

class EnvMap(object):
    """EnvMap object used to load and render map from file"""

    def __init__(self, map_path):
        super(EnvMap, self).__init__()
        self.map_path = map_path
        self.load_map()

    def get_all_treasure_locations(self):
        treasure_locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_treasure(row, col):
                    treasure_locations.append((row, col))
        return treasure_locations

    def get_all_wall_locations(self):
        wall_locations = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.has_wall(row, col):
                    wall_locations.append((row, col))
        return wall_locations

    def get_all_lightning_probability(self):
        lightning_probability = []
        for row in range(len(self.grid)):
            r = []
            for col in range(len(self.grid[row])):
                r.append(self.get_lightning_probability(row, col))
            lightning_probability.append(r)
        return lightning_probability

    def load_map(self):
        self.grid = []

        with open(self.map_path) as fp:
            line = fp.readline()
            while line:
                row = list(map(float, line.strip().split()))
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
        return self.grid[row][col] == 2

    def has_treasure(self, row, col):
        return self.grid[row][col] == 3

    def is_agent_position(self, row, col):
        return self.grid[row][col] == 4

    def get_lightning_probability(self, row, col):
        if (self.has_wall(row, col)
            or self.is_agent_position(row, col)
                or self.has_treasure(row, col)):
            return 0

        return self.grid[row][col]

    def shape(self):
        return (len(self.grid), len(self.grid[0]))
