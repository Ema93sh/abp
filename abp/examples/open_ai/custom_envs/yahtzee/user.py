import gym
import numpy as np
import abp.custom_envs

import time
import curses

screen = curses.initscr()
curses.savetty()
curses.noecho()
curses.cbreak()
curses.curs_set(0)
# screen.nodelay(0)
screen.keypad(1)

def run_task(config):
    env_spec = gym.make("Yahtzee-v0")
    max_episode_steps = env_spec._max_episode_steps

    state = env_spec.reset()

    for epoch in range(config.test_episodes):
        state = env_spec.reset()
        reward = 0
        for steps in range(max_episode_steps):
            screen.clear()
            s = env_spec.render(mode='ansi')
            screen.addstr(s.getvalue())
            screen.refresh()
            category = np.random.choice(range(0, 13))
            action = ([0] * 5, category)
            state, reward, done, info = env_spec.step(action)
            screen.addstr("Action:" + str(action) + "\n")
            key = screen.getch()

            if done:
                screen.clear()
                break

    env_spec.close()
