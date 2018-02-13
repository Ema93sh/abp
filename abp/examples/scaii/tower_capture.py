import copy
import sys
import logging
logger = logging.getLogger('root')

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

    choose_tower = DQNAdaptive(name = "tower", choices = [TOWER_BR, TOWER_BL, TOWER_TR, TOWER_TL], network_config = network_config, reinforce_config = reinforce_config)


    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = tf.summary.FileWriter(training_summaries_path)

    #Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()


        tower_to_kill, _ = choose_tower.predict(state.state)

        action = env.new_action()

        action.attack_quadrant(tower_to_kill)

        state = env.act(action)

        while not state.is_terminal():
            noop = env.new_action()

            state = env.act(noop)

            choose_tower.reward(state.reward)

            total_reward += state.reward

            if state.is_terminal():
                logger.info("End Episode of episode %d!" % (episode + 1))


        choose_tower.end_episode(state.state)

        logging.info("Episode %d : %d" % (episode + 1, total_reward))
        episode_summary.value.add(tag = "Reward", simple_value = total_reward)
        train_summary_writer.add_summary(episode_summary, episode + 1)

    train_summary_writer.flush()

    choose_tower.disable_learning()

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = tf.summary.FileWriter(test_summaries_path)

    #Test Episodes
    if evaluation_config.render:
        env.load_rpc_module("viz")

    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()

        tower_to_kill, _ = choose_tower.predict(state.state)

        logger.info("Attacking Tower...%d" % tower_to_kill)


        action = env.new_action()

        action.attack_quadrant(tower_to_kill)


        state = env.act(action)

        while not state.is_terminal():
            noop = env.new_action()
            state = env.act(noop)
            total_reward += state.reward

            if state.is_terminal():
                logger.info("End Episode of episode %d!" % (episode + 1))
                logger.info("Total Reward %d!" % (total_reward))

    test_summary_writer.flush()
