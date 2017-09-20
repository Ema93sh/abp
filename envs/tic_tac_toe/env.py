import sys
import numpy as np
from six import StringIO

import gym
from gym import spaces

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']} #TODO render to RGB

    # difficulty
    # 0 - random
    # 1 - medium
    # 2 - impossible

    def __init__(self, difficulty = 1):
        self.action_space = spaces.Discrete(9)

        values = np.array([1] * 9)
        self.observation_space = spaces.Box(-values, values)

        self.state = np.array([0]*9)

    def _step(self, action):
        done = False
        reward = 0
        info = {'x_won': False, 'o_won':False, 'illegal_move': False}

        if self.state[action] != 0: # Illegal Move
            done = True
            reward = -20
            info['illegal_move'] = True
        else:
            info['illegal_move'] = False
            self.state[action] = 1

            reward, done = self.check_if_done(info)

            possible_moves = self.get_all_possible_moves()

            if len(possible_moves) == 0:
                reward, done = 20,  True

            if not done:
                reward = 1
                action = self.make_move(possible_moves)
                self.state[action] = -1

        return self.state, reward, done, info

    def check_if_done(self, info):
        reward = 0
        done = False

        reshaped_board = np.reshape(self.state, (3,3))

        sum_rows = np.sum(reshaped_board, axis = 1)
        sum_cols = np.sum(reshaped_board, axis = 0)
        sum_diagonal = np.trace(reshaped_board)
        sum_rev_diagonal = np.trace(np.flipud(reshaped_board))

        sums = np.hstack([sum_rows, sum_cols, sum_diagonal, sum_rev_diagonal])

        if 3 in sums:
            reward, done = 100, True
            info['x_won'] = True
            info['o_won'] = False
        elif -3 in sums:
            reward, done = -1, True
            info['x_won'] = False
            info['o_won'] = True
        else:
            reward, done = 0, False
            info['x_won'] = False
            info['o_won'] = False

        return reward, done

    def make_move(self, possible_moves):
        return np.random.choice(possible_moves)

    def get_all_possible_moves(self):
        return np.where(self.state == 0)[0]

    def _reset(self):
        self.state = np.array([0]*9)
        return self.state

    def _render(self, mode = 'human', close = False):
        if close:
            # Nothing interesting to close
            return

        reshaped_board = np.reshape(self.state, (3,3))

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for i in range(3):
            for j in range(3):
                if reshaped_board[i, j] == 0:
                    outfile.write(' - ')
                elif reshaped_board[i, j] == 1:
                    outfile.write(' X ')
                else:
                    outfile.write(' O ')
            outfile.write('\n')

        outfile.write('\n')

        return outfile
