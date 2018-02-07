import logging
import copy

import gym
import tensorflow as tf

from abp import DQNAdaptive
from abp.utils import clear_summary_path


# Only chosing the category

def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = 10000
    state = env.reset()

    category_choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    which_category = DQNAdaptive(name = "which_category",
                                 choices = category_choices,
                                 network_config = category_network_config,
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

        for step in range(13):

            #Roll Dice Three times
            for dice_step in range(3):
                
                action  = ([dice1_action, dice2_action, dice3_action, dice4_action, dice5_action], None)

                state, reward, done, info = env.step(action)

            #Select Category
            category, _ =  which_category.predict(state)

            action  = ([], category)

            state, reward, done, info = env.step(action)

            dice1.reward(reward)
            dice2.reward(reward)
            dice3.reward(reward)
            dice4.reward(reward)
            dice5.reward(reward)

            which_category.reward(reward)

            total_reward += reward

            if done:
                logging.info("Episode %d : %d" % (episode + 1, total_reward))

                dice1.end_episode(state)
                dice2.end_episode(state)
                dice3.end_episode(state)
                dice4.end_episode(state)
                dice5.end_episode(state)

                which_category.end_episode(state)

                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                train_summary_writer.add_summary(episode_summary, episode + 1)
                break

    train_summary_writer.flush()

    dice1.disable_learning()
    dice2.disable_learning()
    dice3.disable_learning()
    dice4.disable_learning()
    dice5.disable_learning()

    which_category.disable_learning()

    #Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()

        for step in range(13):

            #Roll Dice Three times
            for dice_step in range(3):
                dice1_action, _ = dice1.predict(state)
                dice2_action, _ = dice2.predict(state)
                dice3_action, _ = dice3.predict(state)
                dice4_action, _ = dice4.predict(state)
                dice5_action, _ = dice5.predict(state)

                action  = ([dice1_action, dice2_action, dice3_action, dice4_action, dice5_action], None)

                state, reward, done, info = env.step(action)

            #Select Category
            category, _ =  which_category.predict(state)

            action  = ([], category)

            state, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                test_summary_writer.add_summary(episode_summary, episode + 1)
                break

    test_summary_writer.flush()
