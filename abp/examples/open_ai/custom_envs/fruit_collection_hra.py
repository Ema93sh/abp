import gym
import numpy as np
import abp.custom_envs
import os
import time

from abp.adaptives.hra import HRAAdaptive

def run_task(job_dir, render = True, training_episode = 2000, test_episodes = 100, decay_steps = 250,  model_path = None, restore_model = False):
    env_spec = gym.make("FruitCollection-v0")
    max_episode_steps = env_spec._max_episode_steps


    no_of_rewards = 10

    state = env_spec.reset()
    agent = HRAAdaptive(env_spec.action_space.n,
                        len(state),
                        no_of_rewards,
                        "Fruit Collection",
                        job_dir = job_dir,
                        decay_steps = decay_steps,
                        gamma = 0.95,
                        model_path = model_path,
                        restore_model = restore_model)


    #Training Episodes
    for epoch in range(training_episode):
        state = env_spec.reset()
        for steps in range(max_episode_steps):

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


            agent.actual_reward(-1)


            if done or steps == (max_episode_steps - 1):
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
