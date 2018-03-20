import gym
import numpy as np
import abp
import time

env_spec = gym.make("FruitCollection-v0")
max_episode_steps = 300

import curses

screen = curses.initscr()
curses.savetty()
curses.noecho()
curses.cbreak()
curses.curs_set(0)
# screen.nodelay(0)
screen.keypad(1)



def get_arrow_key():
        LEFT, RIGHT, UP, DOWN, NOOP = [0, 1, 2, 3, 4]
        k = screen.getch()

        if k == curses.KEY_UP:
            screen.addstr("Up")
            return UP
        elif k == curses.KEY_DOWN:
            screen.addstr("Down")
            return DOWN
        elif k == curses.KEY_RIGHT:
            screen.addstr("Right")
            return RIGHT
        elif k == curses.KEY_LEFT:
            screen.addstr("Left")
            return LEFT

        return get_arrow_key()

# Test Episodes
for epoch in range(100):
    reward = 0
    state = env_spec.reset(state_mode = "grid")
    for steps in range(max_episode_steps):
        screen.clear()
        s = env_spec.render(mode='ansi')
        screen.addstr(s.getvalue())
        screen.addstr(str(state))
        screen.refresh()
        action = get_arrow_key()

        state, reward, done, info = env_spec.step(action)

        r = None

        if done or steps == (max_episode_steps - 1):
            break

env_spec.close()
