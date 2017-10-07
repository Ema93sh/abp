import gym
import numpy as np
import abp.custom_envs
import os
import time

from abp.adaptives.hra import HRAAdaptive

def run_task(config):
    config.name = "FruitCollection-v0"

    env_spec = gym.make(config.name)
    state = env_spec.reset()
    max_episode_steps = env_spec._max_episode_steps

    config.size_rewards = 10
    config.size_features = len(state)
    config.action_size = env_spec.action_space.n

    agent = HRAAdaptive(config)

    #Training Episodes
    for epoch in range(config.training_episode):
        state = env_spec.reset()
        for steps in range(max_episode_steps):

            action = agent.predict(state)
            state, reward, done, info = env_spec.step(action)

            possible_fruit_locations = info["possible_fruit_locations"]
            collected_fruit = info["collected_fruit"]
            current_fruit_locations = info["current_fruit_locations"]

            r = None
            if collected_fruit is not None:
                r = possible_fruit_locations.index(collected_fruit)
                agent.reward(r, 1)

            for i in range(9):
                if (r is None or r != i) and  possible_fruit_locations[i] in current_fruit_locations:
                    agent.reward(i, -1)

            agent.actual_reward(-1)


            if done or steps == (max_episode_steps - 1):
                agent.end_episode(state)
                break

    agent.disable_learning()

    if config.render: #TODO Move inside ENV
        import curses
        screen = curses.initscr()
        curses.savetty()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        #Test Episodes
        for epoch in range(config.test_episodes):
            state = env_spec.reset()
            for steps in range(max_episode_steps):
                screen.clear()
                screen.addstr("Episode: " + str(agent.episode) + "\n")
                screen.addstr("Reward: " + str(agent.total_psuedo_reward) + "\n")
                screen.addstr("Step: " + str(steps) + "\n")
                s = env_spec.render(mode = "ansi")
                screen.addstr(s.getvalue())

                action = agent.predict(state)
                screen.addstr("Action: " +str(action) + "\n")

                screen.refresh()
                time.sleep(1)

                state, reward, done, info = env_spec.step(action)
                agent.test_reward(-1)

                if done:
                    agent.end_episode(state)
                    break
    else:
        #Test Episodes
        for epoch in range(test_episodes):
            state = env_spec.reset()
            for steps in range(max_episode_steps):
                if render:
                    env_spec.render()
                action = agent.predict(state)
                state, reward, done, info = env_spec.step(action)
                agent.test_reward(-1)

                if done:
                    agent.end_episode(state)
                    break


    env_spec.close()
