import gym
import numpy as np
import abp.custom_envs

from abp.adaptives.dqn import DQNAdaptive


def run_task(job_dir, render = True, training_episode = 2000, test_episodes = 100, decay_steps = 500, model_path = None, restore_model = False):
    env_spec = gym.make("FruitCollection-v0")
    max_episode_steps = env_spec._max_episode_steps

    state = env_spec.reset()
    agent = DQNAdaptive(env_spec.action_space.n,
                        len(state), "Fruit Collection",
                        job_dir = job_dir,
                        decay_steps = decay_steps,
                        model_path = model_path,
                        restore_model = restore_model)

    #Training Episodes
    for epoch in range(training_episode):
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            action = agent.predict(state)
            state, reward, done, info = env_spec.step(action)

            agent.reward(reward)

            agent.actual_reward(-1)

            if done or steps == (max_episode_steps - 1):
                agent.end_episode(state)
                break

    agent.disable_learning()

    #Test Episodes
    for epoch in range(test_episodes):
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            if render:
                env_spec.render()
            action = agent.predict(state)
            state, reward, done, info = env_spec.step(action)
            agent.test_reward(reward)

            if done:
                agent.end_episode(state)
                break

    env_spec.close()
