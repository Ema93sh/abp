import gym
import numpy as np
import abp.custom_envs
import os
import time


def run_task(config):
    env_spec = gym.make("Traveller-v0")
    max_episode_steps = env_spec._max_episode_steps

    import curses

    screen = curses.initscr()
    curses.savetty()
    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)
    # screen.nodelay(0)
    screen.keypad(1)


    # Test Episodes
    for epoch in range(config.test_episodes):
        reward = 0
        days_remaining = 8
        total_reward = 0
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            screen.clear()
            s = env_spec.render(mode='ansi')
            screen.addstr(s.getvalue())
            screen.addstr("Days Remaining: " + str(days_remaining) + "\n")
            screen.addstr("Reward: " + str(reward) + "\n")
            screen.addstr("Total Reward: " + str(total_reward) + "\n")
            screen.addstr("State: " + str(state) + "\n")
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

            state, reward, done, info = env_spec.step(action)
            total_reward += reward
            days_remaining = info['days_remaining']
            screen.refresh()


            if done or steps == (max_episode_steps - 1):
                screen.addstr("Reward: " + str(reward) + "\n")
                screen.addstr("Total Reward: " + str(total_reward) + "\n")
                screen.addstr("End Episode \n")
                screen.refresh()
                time.sleep(1)
                break

    env_spec.close()
