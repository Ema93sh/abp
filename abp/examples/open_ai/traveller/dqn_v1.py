import sys

import gym
import numpy as np
import tensorflow as tf

from abp import DQNAdaptive
from abp.utils import clear_summary_path
from abp.utils.histogram import SingleQHistogram

def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = env._max_episode_steps
    state = env.reset()

    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]

    traveller = DQNAdaptive(name="traveller",
                        choices = [LEFT, RIGHT, UP, DOWN],
                        network_config = network_config,
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
            action, _ = traveller.predict(state)
            state, reward, done, info = env.step(action)
            total_reward += reward

            traveller.reward(reward)

            if done:
                traveller.end_episode(state)

                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                train_summary_writer.add_summary(episode_summary, episode + 1)
                break


    traveller.disable_learning()

    #Test Episodes
    # TODO
    chart = SingleQHistogram(env.action_space.n, ('Left', 'Right', 'Up', 'Down'),  y_lim = 20)


    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        reward = 0
        action = None
        state = env.reset()
        days_remaining = 8
        total_reward = 0
        episode_summary = tf.Summary()
        for steps in range(max_episode_steps):
            action, q_values = traveller.predict(state)

            if evaluation_config.render:
                s = env.render(mode='ansi')
                print(s.getvalue())
                chart.render(q_values)
                print("Press enter to continue:")
                sys.stdin.read(1)

            state, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                if evaluation_config.render:
                    s = env.render(mode='ansi')
                    print(s.getvalue())
                    print("********** END OF EPISODE *********")
                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                test_summary_writer.add_summary(episode_summary, episode + 1)
                break

    env.close()
