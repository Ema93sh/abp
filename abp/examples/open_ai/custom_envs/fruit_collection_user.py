import gym
import numpy as np
import abp.custom_envs
import os
import time

from abp.adaptives.hra import HRAAdaptive

def run_task(job_dir, render = True, training_episode = 100, test_episodes = 100, decay_steps = 250,  model_path = None, restore_model = False):
    env_spec = gym.make("FruitCollection-v0")
    max_episode_steps = env_spec._max_episode_steps



    import curses

    screen = curses.initscr()
    curses.savetty()
    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)
    # screen.nodelay(0)
    screen.keypad(1)

    #Training Episodes
    for epoch in range(training_episode):
        reward = 0
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            screen.clear()
            screen.addstr("Reward: " + str(reward) + "\n")
            screen.addstr("State:\n" + str(state) + "\n\n")
            screen.refresh()
            s = env_spec.render(mode = "ansi")
            screen.addstr(s.getvalue())
            screen.refresh()
            key = screen.getch()
            action = 0

            if key ==  259:
                action = 2
            elif key == 258:
                action = 3
            elif key == 260:
                action = 0
            elif key == 261:
                action = 1

            state, reward, done, info = env_spec.step(action)

            possible_fruit_locations = info["possible_fruit_locations"]
            collected_fruit = info["collected_fruit"]

            if done or steps == (max_episode_steps - 1):
                break

    env_spec.close()
