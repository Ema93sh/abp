import gym
import numpy as np
import tensorflow as tf

from abp import HRAAdaptive
from abp.utils import clear_summary_path
from abp.openai.wrappers import RewardWrapper

#TODO
# *reward wrapper

def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = env._max_episode_steps
    # env = RewardWrapper(env)
    state = env.reset()
    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]

    agent = HRAAdaptive(name = "FruitCollecter",
                        choices = [LEFT, RIGHT, UP, DOWN],
                        network_config =  network_config,
                        reinforce_config = reinforce_config)

    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = tf.summary.FileWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = tf.summary.FileWriter(test_summaries_path)

    #Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()
        for steps in range(max_episode_steps):
            action, q_values = agent.predict(state)
            state, reward, done, info = env.step(action)

            agent.reward(reward)

            total_reward += sum(reward)

            if done or steps == (max_episode_steps - 1):
                agent.end_episode(state)
                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                episode_summary.value.add(tag = "Steps to collect all fruits",  simple_value = steps + 1)
                train_summary_writer.add_summary(episode_summary, episode + 1)
                break

    train_summary_writer.flush()

    agent.disable_learning()

    #Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()

        for steps in range(max_episode_steps):
            action, q_values = agent.predict(state)

            state, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                agent.end_episode(state)
                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                episode_summary.value.add(tag = "Steps to collect all fruits", simple_value = steps + 1)
                test_summary_writer.add_summary(episode_summary, episode + 1)
                break

    test_summary_writer.flush()
    env.close()
