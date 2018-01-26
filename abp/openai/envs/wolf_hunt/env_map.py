class EnvMap(object):
    """EnvMap object used to load and render map from file"""
    WALL, RABBIT, WOLF_1, WOLF_2 = [1, 2, 3, 4]

    def __init__(self, map_path):
        super(EnvMap, self).__init__()
        self.map_path = map_path
        self.grid = []
        self.load_map()
        self.wolf1_location = self.get_location(self.WOLF_1)
        self.wolf2_location = self.get_location(self.WOLF_2)
        self.rabbit_location = self.get_location(self.RABBIT)


    def get_location(self, loc_type):
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.grid[row][col] == loc_type:
                    return (row, col)
        return None


    def load_map(self):
        self.grid = []

        with open(self.map_path) as fp:
            line = fp.readline()
            while line:
                row = map(int, line.strip().split(' '))
                self.grid.append(row)
                line = fp.readline()


    def render(self):
        for row in self.grid:
            print row


    def has_wall(self, row, col):
        return self.grid[row][col] == self.WALL


    def size(self):
        return len(self.grid), len(self.grid[0])
