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
max_episode_steps = 1000

state = env_spec.reset()
predators = env_spec.predators

def get_arrow_key():
        LEFT, RIGHT, UP, DOWN, NOOP = [0, 1, 2, 3, 4]
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
    state = env_spec.reset(map_name = "5x5_default")
    reward = 0
    for steps in range(100):
        screen.clear()
        s = env_spec.render(mode='ansi')
        screen.addstr(s.getvalue())
        screen.addstr("State\n")
        screen.addstr(str(state))
        actions = {}

        for predator in predators:
            screen.addstr("Direction for %s:\n" % predator)
            screen.refresh()
            action = get_arrow_key()
            actions[predator] = action

        state, reward, done, info = env_spec.step(actions)


        if done:
            screen.clear()
            break

env_spec.close()
