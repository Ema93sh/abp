import gym
import numpy as np
import abp

import os
import time


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

def decompose_reward(state_info, done):
    #TODO Move this as part of the environment!!! SOOOOO MESSYYYY! USE A WRAPPER FOR ENV!
    traveller_location = state_info["traveller_location"]

    reward_map = {
      "HOME"     : 0,
      "TREASURE" : 0,
      "TERRAIN"  : 0
    }


    if traveller_location in state_info["mountain_locations"]:
        reward_map["TERRAIN"] -= 4

    if traveller_location in state_info["hill_locations"]:
        reward_map["TERRAIN"] -= 3

    if traveller_location in state_info["river_locations"]:
        reward_map["TERRAIN"] -= 2

    if traveller_location in state_info["gold_locations"]:
        reward_map["TREASURE"] += 2
    if traveller_location in state_info["diamond_locations"]:
        reward_map["TREASURE"] += 3

    if done and traveller_location == state_info["house_location"]:
        reward_map["HOME"] += 10

    if done and traveller_location == state_info["house_location"]:
        reward_map["HOME"] -= 10

    return reward_map

# Test Episodes
for epoch in range(100):
    reward = 0
    days_remaining = 8
    total_reward = 0
    info = None
    d_reward = []
    state = env_spec.reset()
    for steps in range(max_episode_steps):
        screen.clear()
        s = env_spec.render(mode='ansi')
        screen.addstr(s.getvalue())
        screen.addstr("Days Remaining: %d\n" % days_remaining)
        screen.addstr("Reward: %d \n" % reward)
        if info:
            screen.addstr("DReward: %s \n" % str(info["decomposed_reward"]))
            if reward != sum(info["decomposed_reward"].values()):
                print("FACKKK!")
                exit(0)
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

        state, reward, done, info = env_spec.step(action)
        total_reward += reward

        days_remaining = info['days_remaining']
        screen.refresh()


        if done or steps == (max_episode_steps - 1):
            screen.clear()
            s = env_spec.render(mode='ansi')
            screen.addstr(s.getvalue())
            screen.addstr("Reward: " + str(reward) + "\n")
            screen.addstr("DReward: %s \n" % str(info["decomposed_reward"]))
            screen.addstr("Total Reward: " + str(total_reward) + "\n")
            screen.addstr("End Episode \n")
            screen.refresh()
            time.sleep(5)
            break

env_spec.close()
