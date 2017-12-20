import gym
import numpy as np
import abp
import os
import time

from abp import HRAAdaptive
from abp.utils.bar_chart import MultiQBarChart

def run_task(config):
    config.name = "FruitCollection-v0"

    env_spec = gym.make(config.name)
    state = env_spec.reset()
    max_episode_steps = env_spec._max_episode_steps

    config.size_rewards = 10
    config.size_features = len(state)
    config.action_size = env_spec.action_space.n

    agent = HRAAdaptive(config)
    possible_fruit_locations = env_spec.env.possible_fruit_locations

    #Training Episodes
    for epoch in range(config.training_episode):
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            action, q_values = agent.predict(state)
            state, reward, done, info = env_spec.step(action)

            possible_fruit_locations = info["possible_fruit_locations"]
            collected_fruit = info["collected_fruit"]
            current_fruit_locations = info["current_fruit_locations"]

            r = None
            if collected_fruit is not None:
                r = possible_fruit_locations.index(collected_fruit)
                agent.reward(r, 1, "Collected Fruit at location %d" % collected_fruit)

            # for i in range(9):
            #     if (r is None or r != i) and  possible_fruit_locations[i] in current_fruit_locations:
            #         agent.reward(i, -1, "Missed Fruit at location %d" % possible_fruit_locations[i])

            agent.actual_reward(-1)


            if done or steps == (max_episode_steps - 1):
                agent.end_episode(state)
                break

    agent.disable_learning()

    chart = MultiQBarChart(config.size_rewards, env_spec.action_space.n, ('Left', 'Right', 'Up', 'Down'), width = 0.08)

    import curses

    screen = curses.initscr()
    curses.savetty()
    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)
    # screen.nodelay(0)
    screen.keypad(1)

    #Test Episodes
    for epoch in range(config.test_episodes):
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            # env_spec.render()
            screen.clear()
            s = env_spec.render(mode='ansi')
            screen.addstr(s.getvalue())
            action, q_values = agent.predict(state)
            merged = np.sum(q_values, axis=0)
            merged_action = np.argmax(merged)
            chart.render(q_values, [ "Location_(%d, %d)" % ((possible_fruit_locations[i] / 10 + 1), (possible_fruit_locations[i] % 10 + 1)) for i in range(config.size_rewards)])
            screen.refresh()

            time.sleep(10)
            state, reward, done, info = env_spec.step(action)
            agent.test_reward(-1)

            if done:
                agent.end_episode(state)
                break

    env_spec.close()
