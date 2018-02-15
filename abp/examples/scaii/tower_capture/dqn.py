import copy
import sys
import logging
logger = logging.getLogger('root')

import time
from scaii.env.sky_rts.env.scenarios.tower_example import TowerExample
import tensorflow as tf
import numpy as np

from abp import DQNAdaptive
from abp.utils import clear_summary_path


# Size State: 100x100x6

def run_task(evaluation_config, network_config, reinforce_config):
    env = TowerExample()

    max_episode_steps = 10000

    state = env.reset()

    TOWER_BR, TOWER_BL, TOWER_TR, TOWER_TL = [1, 2, 3, 4]

    choose_tower = DQNAdaptive(name = "tower",
                               choices = [TOWER_BR, TOWER_BL, TOWER_TR, TOWER_TL],
                               network_config = network_config,
                               reinforce_config = reinforce_config)


    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = tf.summary.FileWriter(training_summaries_path)

    #Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()

        start_time = time.time()
        tower_to_kill, _ = choose_tower.predict(state.state)
        end_time = time.time()

        action = env.new_action()

        env_start_time = time.time()
        action.attack_quadrant(tower_to_kill)

        state = env.act(action)

        counter = 0

        choose_tower.reward(state.reward)

        total_reward += state.reward

        if state.is_terminal():
            logger.info("End Episode of episode %d!" % (episode + 1))
            logger.info("Total Reward %d!" % (total_reward))

        env_end_time = time.time()

        logger.debug("Counter: %d" % counter)
        logger.debug("Neural Network Time: %.2f" % (end_time - start_time))
        logger.debug("Env Time: %.2f" % (env_end_time - env_start_time))

        choose_tower.end_episode(state.state)

        episode_summary.value.add(tag = "Reward", simple_value = total_reward)
        train_summary_writer.add_summary(episode_summary, episode + 1)

    train_summary_writer.flush()

    choose_tower.disable_learning()

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = tf.summary.FileWriter(test_summaries_path)

    #Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset(visualize=evaluation_config.render)
        total_reward = 0
        episode_summary = tf.Summary()

        tower_to_kill, _ = choose_tower.predict(state.state)

        action = env.new_action()

        action.attack_quadrant(tower_to_kill)

        state = env.act(action)

        total_reward += state.reward

        if state.is_terminal():
            logger.info("End Episode of episode %d!" % (episode + 1))
            logger.info("Total Reward %d!" % (total_reward))

        episode_summary.value.add(tag = "Reward", simple_value = total_reward)
        test_summary_writer.add_summary(episode_summary, episode + 1)

    test_summary_writer.flush()
