import os
from itertools import chain

from six import StringIO
import gym

from env_map import EnvMap
from rabbit import Rabbit

#TODO WTF MAN! Write Tests!!!

class WolfHuntEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    WOLF_1, WOLF_2, RABBIT  = [0, 1, 2]
    UP, DOWN, LEFT, RIGHT, NOOP = [0, 1, 2, 3, 4]

    "Two wolves hunting a rabbit."

    def __init__(self, env_map_path = None):
        super(WolfHuntEnv, self).__init__()

        if env_map_path is None:
            env_map_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "maps/5x5_simple.map")

        self.env_map = EnvMap(env_map_path)
        self.rabbit = Rabbit()

        self._reset()


    def _reset(self):
        self.current_location = {
           self.RABBIT : self.env_map.rabbit_location,
           self.WOLF_1 : self.env_map.wolf1_location,
           self.WOLF_2 :  self.env_map.wolf2_location
        }
        return self.generate_state()

    def generate_grid(self):
        row, col = self.env_map.size()
        return [[0 for j in range(col)]for i in range(row)]

    def generate_state(self):

        def flatten(l):
            return [ item for sublist in l for item in sublist]

        rabbit_grid = self.generate_grid()
        row, col = self.current_location[self.RABBIT]
        rabbit_grid[row][col] = 1

        wolf1_grid = self.generate_grid()
        row, col = self.current_location[self.WOLF_1]
        wolf1_grid[row][col] = 1

        wolf2_grid = self.generate_grid()
        row, col = self.current_location[self.WOLF_2]
        wolf2_grid[row][col] = 1

        return list(chain.from_iterable(map(flatten, [rabbit_grid, wolf1_grid, wolf2_grid])))


    def is_done(self):
        if self.current_location[self.RABBIT] == self.current_location[self.WOLF_1]:
            return True, 10

        if self.current_location[self.RABBIT] == self.current_location[self.WOLF_2]:
            return True, 10

        return False, 0


    def next_location(self, current_location, direction):
        row, col = current_location
        if direction == self.UP:
            row -= 1

        if direction == self.DOWN:
            row += 1

        if direction == self.LEFT:
            col -= 1

        if direction == self.RIGHT:
            col += 1

        return (row, col)


    def move(self, piece, action):
        if action == self.NOOP:
            return

        location = self.current_location[piece]
        updated_location = self.next_location(location, action)

        if self.env_map.has_wall(*updated_location):
            return

        if piece == self.WOLF_1 and updated_location == self.current_location[self.WOLF_2]:
            return

        if piece == self.WOLF_2 and updated_location == self.current_location[self.WOLF_1]:
            return

        self.current_location[piece] = updated_location


    def _step(self, action):
        info  = {}

        wolf1_move, wolf2_move = action

        self.move(self.WOLF_1, wolf1_move)

        self.move(self.WOLF_2, wolf2_move)

        rabbit_move = self.rabbit.random_move()

        self.move(self.RABBIT, rabbit_move)

        done, reward = self.is_done()

        return self.generate_state(), reward, done, info

    def render_ansi(self):
        output = StringIO()
        row, col = self.env_map.size()
        done = (self.current_location[self.RABBIT] == self.current_location[self.WOLF_1]) or (self.current_location[self.RABBIT] == self.current_location[self.WOLF_2])

        for i in range(row):
            for j in range(col):
                if done and (i, j) == self.current_location[self.RABBIT]:
                    output.write("  X ")
                elif (i, j) == self.current_location[self.WOLF_1]:
                    output.write(" W1 ")
                elif (i, j) == self.current_location[self.WOLF_2]:
                    output.write(" W2 ")
                elif (i, j) == self.current_location[self.RABBIT]:
                    output.write("  R ")
                elif self.env_map.has_wall(i, j):
                    output.write("  - ")
                else:
                    output.write(" .  ")
            output.write("\n")

        return output


    def _render(self, mode = 'ansi', close = False):
        if close:
            return None

        return self.render_ansi()
