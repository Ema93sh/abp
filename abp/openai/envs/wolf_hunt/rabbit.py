import numpy as np

#TODO Make it more intelligent


class Rabbit(object):
    """AI for the rabbit"""
    UP, DOWN, LEFT, RIGHT, NOOP = [0, 1, 2, 3, 4]

    def __init__(self):
        super(Rabbit, self).__init__()

    def random_move(self):
        moves = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.NOOP]
        return np.random.choice(moves)
