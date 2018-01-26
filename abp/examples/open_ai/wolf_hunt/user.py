import gym

import abp

import time
import curses

screen = curses.initscr()
curses.savetty()
curses.noecho()
curses.cbreak()
curses.curs_set(0)
# screen.nodelay(0)
screen.keypad(1)

env_spec = gym.make("WolfHunt-v0")
max_episode_steps = env_spec._max_episode_steps

state = env_spec.reset()


def get_arrow_key():
        UP, DOWN, LEFT, RIGHT, NOOP = [0, 1, 2, 3, 4]
        k = screen.getch()

        if k == curses.KEY_UP:
            return UP
        elif k == curses.KEY_DOWN:
            return DOWN
        elif k == curses.KEY_RIGHT:
            return RIGHT
        elif k == curses.KEY_LEFT:
            return LEFT

        return get_arrow_key()

for epoch in range(10):
    state = env_spec.reset()
    reward = 0
    for steps in range(100):
        screen.clear()
        s = env_spec.render(mode='ansi')
        screen.addstr("State\n")
        # screen.addstr(str(state))
        screen.addstr(s.getvalue())
        screen.addstr("Direction for Wolf 1:\n")
        screen.refresh()

        wolf1 = get_arrow_key()
        screen.addstr("Direction for Wolf 2:\n")
        screen.refresh()

        wolf2 = get_arrow_key()

        action = (wolf1, wolf2)
        state, reward, done, info = env_spec.step(action)


        if done:
            screen.clear()
            break

env_spec.close()
