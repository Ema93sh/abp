import copy
import sys
import logging

import gym
import tensorflow as tf

from abp import DQNAdaptive, QPredictor
from abp.utils import clear_summary_path

def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = 10000
    state = env.reset()
    UP, DOWN, LEFT, RIGHT, NOOP = [0, 1, 2, 3, 4]

    wolf1 = DQNAdaptive(name = "wolf1", choices = [UP, DOWN, LEFT, RIGHT, NOOP], network_config = network_config, reinforce_config = reinforce_config)
    wolf2 = DQNAdaptive(name = "wolf2", choices = [UP, DOWN, LEFT, RIGHT, NOOP], network_config = network_config, reinforce_config = reinforce_config)

    steps_to_catch_rabbit = QPredictor(name = "No Steps to catch rabbit", network_config = network_config)

    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = tf.summary.FileWriter(training_summaries_path)

    #Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()

        for step in range(max_episode_steps):
            wolf1_action, _ = wolf1.predict(state)
            wolf2_action, _ = wolf2.predict(state)

            actions = {}
            actions["W1"] = wolf1_action
            actions["W2"] = wolf2_action

            state, reward, done, info = env.step(actions)

            steps_to_catch_rabbit.learn(state, wolf1_action, 1, done, 0)

            wolf1.reward(reward)
            wolf2.reward(reward)

            total_reward += reward

            if done:
                wolf1.end_episode(state)
                wolf2.end_episode(state)

                logging.info("Episode %d : %d" % (episode + 1, total_reward))
                episode_summary.value.add(tag = "Reward", simple_value = total_reward)
                episode_summary.value.add(tag = "Steps", simple_value = step + 1)
                train_summary_writer.add_summary(episode_summary, episode + 1)
                break

    train_summary_writer.flush()

    wolf1.disable_learning()
    wolf2.disable_learning()

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = tf.summary.FileWriter(test_summaries_path)


    #Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()
        action = None
        for step in range(max_episode_steps):
            if evaluation_config.render:
                v = steps_to_catch_rabbit.predict(state)
                s = env.render()
                print(s.getvalue())
                print("Steps to catch rabbit: ", v)
                print("Press enter to continue:")
                sys.stdin.read(1)

            wolf1_action, _ = wolf1.predict(state)
            wolf2_action, _ = wolf2.predict(state)

            actions = {}
            actions["W1"] = wolf1_action
            actions["W2"] = wolf2_action

            state, reward, done, info = env.step(actions)

            total_reward += reward

            if evaluation_config.render:
                w1, w2 = action
                print("Wolf1 Action", env.env.action_map[w1])
                print("Wolf2 Action", env.env.action_map[w2])
                print("Reward", reward)

            if done:
                if evaluation_config.render:
                    s = env.render()
                    print(s.getvalue())
                    print("Total Reward", total_reward)
                    print("********** END OF EPISODE *********")
                episode_summary.value.add(tag = "Reward", simple_value = total_reward)
                episode_summary.value.add(tag = "Steps", simple_value = step + 1)
                test_summary_writer.add_summary(episode_summary, episode + 1)
                break

    test_summary_writer.flush()
