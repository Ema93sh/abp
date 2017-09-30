import gym
import numpy as np
import abp.envs
import os
import time

from abp.adaptives.hra import HRAAdaptive

env_spec = gym.make("FruitCollection-v0")
max_episode_steps = env_spec._max_episode_steps


no_of_rewards = 10
training_episode = 2000
test_episodes = 100

state = env_spec.reset()
agent = HRAAdaptive(env_spec.action_space.n, len(state), no_of_rewards, "Fruit Collection", decay_steps = 250, gamma = 0.95)

# import curses
#
# screen = curses.initscr()
# curses.savetty()
# curses.noecho()
# curses.cbreak()
# curses.curs_set(0)
# # screen.nodelay(0)
# screen.keypad(1)

#Training Episodes
for epoch in range(training_episode):
    state = env_spec.reset()
    for steps in range(max_episode_steps):
        # screen.clear()
        # s = env_spec.render(mode = "ansi")
        # screen.addstr(s.getvalue())
        # screen.refresh()
        # key = screen.getch()
        # action = 0
        #
        # if key ==  259:
        #     action = 2
        # elif key == 258:
        #     action = 3
        # elif key == 260:
        #     action = 0
        # elif key == 261:
        #     action = 1

        action = agent.predict(state)
        state, reward, done, info = env_spec.step(action)

        possible_fruit_locations = info["possible_fruit_locations"]
        collected_fruit = info["collected_fruit"]

        r = None
        if collected_fruit is not None:
            r = possible_fruit_locations.index(collected_fruit)
            agent.reward(r, 1)

        for i in range(9):
            if r is None or r != i:
                agent.reward(i, -1)

        # agent.reward(10, -1)

        agent.actual_reward(-1)


        if done or steps == (max_episode_steps - 1):
            # screen.clear()
            # s = env_spec.render(mode = "ansi")
            # screen.addstr(s.getvalue())
            # screen.addstr("\nTAR:" + str(agent.total_actual_reward) + "\n")
            # screen.addstr("\nTPR:" + str(agent.total_psuedo_reward) + "\n")
            # screen.refresh()
            # time.sleep(1)
            agent.end_episode(state)
            break

agent.disable_learning()

#Test Episodes
for epoch in range(test_episodes):
    state = env_spec.reset()
    for steps in range(max_episode_steps):
        env_spec.render()
        action = agent.predict(state)
        state, reward, done, info = env_spec.step(action)
        agent.test_reward(-1)

        if done:
            agent.end_episode(state)
            break


env_spec.close()
