import gym
import numpy as np
import abp.custom_envs
import os
import time

from abp.adaptives.hra import HRAAdaptive

def run_task(config):
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
    for epoch in range(config.test_episodes):
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
            current_fruit_locations = info["current_fruit_locations"]

            r = None
            ps_reward = [0] * 9
            if collected_fruit is not None:
                r = possible_fruit_locations.index(collected_fruit)
                ps_reward[r] =  1
            screen.addstr("R:" + str(r) +"\n")

            for i in range(9):
                if (r is None or r != i) and  possible_fruit_locations[i] in current_fruit_locations:
                    ps_reward[i] = -1

            screen.addstr(str(ps_reward) + "\n")
            screen.refresh()
            time.sleep(1)


            if done or steps == (max_episode_steps - 1):
                break

    env_spec.close()
