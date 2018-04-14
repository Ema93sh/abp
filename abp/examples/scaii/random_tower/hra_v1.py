import copy
import sys
import logging
logger = logging.getLogger('root')

import time
from scaii.env.sky_rts.env.scenarios.random_towers import RandomTowers
import tensorflow as tf
import numpy as np

from abp import HRAAdaptive
from abp.utils import clear_summary_path

from abp.utils.histogram import MultiQHistogram

# Size State: 100x100x6
def decompose_reward(state):
    #TODO This should go inside adaptive

    reward = state.typed_reward

    r_type = {
        'dealt_damage' : 0,
        'agent_death' : 1,
        'bonus' : 2,
        'took_damage': 3
        }

    d_reward = [0, 0, 0, 0, 0]

    for r, v in reward.items():
        d_reward[r_type[r]] = v

    return d_reward


def run_task(evaluation_config, network_config, reinforce_config):
    env = RandomTowers()

    max_episode_steps = 10000

    state = env.reset()

    #TODO generate network config from reward types

    TOWER_LEFT, TOWER_RIGHT = [1, 2]

    choose_tower = HRAAdaptive(name = "tower",
                               choices = [TOWER_LEFT, TOWER_RIGHT],
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
        tower_to_kill, q_values = choose_tower.predict(state.state)
        end_time = time.time()

        action = env.new_action()

        env_start_time = time.time()

        action.attack_tower(tower_to_kill)

        state = env.act(action)

        d_reward = decompose_reward(state)

        choose_tower.reward(d_reward)

        total_reward += state.reward
        env_end_time = time.time()

        logger.debug("Neural Network Time: %.2f" % (end_time - start_time))
        logger.debug("Env Time: %.2f" % (env_end_time - env_start_time))

        choose_tower.end_episode(state.state)

        logger.info("Episode %d : %d" % (episode + 1, total_reward))
        episode_summary.value.add(tag = "Reward", simple_value = total_reward)
        train_summary_writer.add_summary(episode_summary, episode + 1)

    train_summary_writer.flush()

    logger.info("Disabled Learning..")
    choose_tower.disable_learning()

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = tf.summary.FileWriter(test_summaries_path)


    chart = MultiQHistogram(choose_tower.reward_types, len(choose_tower.choices), ("Left Tower","Right Tower"), ylim = 5)

    q_labels = ["Damage Dealt", "Agent Died", "Bonus", "Damage Received"]

    #Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset(visualize=evaluation_config.render)
        total_reward = 0
        episode_summary = tf.Summary()
        tower_to_kill, q_values = choose_tower.predict(state.state)

        if evaluation_config.render:
            chart.render(q_values, q_labels)

        action = env.new_action()
        action.attack_tower(tower_to_kill)
        action.skip = False

        state = env.act(action)

        while not state.is_terminal():
            time.sleep(0.3)
            action = env.new_action()
            action.skip = False
            state = env.act(action)

        total_reward += state.reward

        if state.is_terminal():
            logger.info("End Episode of episode %d!" % (episode + 1))
            logger.info("Total Reward %d!" % (total_reward))

        episode_summary.value.add(tag = "Reward", simple_value = total_reward)
        test_summary_writer.add_summary(episode_summary, episode + 1)

    test_summary_writer.flush()
