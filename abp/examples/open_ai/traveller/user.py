import gym
import numpy as np
import abp

import os
import time


env_spec = gym.make("Traveller-v0")
max_episode_steps = 1000

import curses

screen = curses.initscr()
curses.savetty()
curses.noecho()
curses.cbreak()
curses.curs_set(0)
# screen.nodelay(0)
screen.keypad(1)

# Test Episodes
for epoch in range(100):
    reward = 0
    days_remaining = 8
    total_reward = 0
    info = None
    state = env_spec.reset()
    for steps in range(max_episode_steps):
        screen.clear()
        s = env_spec.render(mode='ansi')
        screen.addstr(s.getvalue())
        screen.addstr("Days Remaining: %d\n" % days_remaining)
        screen.addstr("Reward: %s \n" % str(reward))
        screen.addstr("Total Reward: %d \n" % total_reward)
        screen.addstr("State: %s \n Len State: %d\n" % (str(state),len(state)))
        screen.refresh()
        key = screen.getch()
        action = None

        if key ==  259:
            action = 2
        elif key == 258:
            action = 3
        elif key == 260:
            action = 0
        elif key == 261:
            action = 1

        state, reward, done, info = env_spec.step(action, decompose_level = 1)
        total_reward += sum(reward)

        days_remaining = info['days_remaining']
        screen.refresh()


        if done or steps == (max_episode_steps - 1):
            screen.clear()
            s = env_spec.render(mode='ansi')
            screen.addstr(s.getvalue())
            screen.addstr("Reward: " + str(reward) + "\n")
            screen.addstr("Total Reward: " + str(total_reward) + "\n")
            screen.addstr("End Episode \n")
            screen.refresh()
            time.sleep(5)
            break

env_spec.close()
