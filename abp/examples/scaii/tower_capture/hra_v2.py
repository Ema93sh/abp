import copy
import sys
import logging
logger = logging.getLogger('root')

import time
from scaii.env.sky_rts.env.scenarios.tower_example import TowerExample
from scaii.env.explanation import Explanation, BarChart, BarGroup, Bar
import tensorflow as tf
import numpy as np

from abp import HRAAdaptive
from abp.utils import clear_summary_path

from abp.utils.histogram import MultiQHistogram

# Size State: 100x100x6
def decompose_reward(env, state):
    r_type = {
        'agent_death' : 0,
        'enemy_kill'  : 1,
        'friendly_kill': 2
    }

    d_reward = [0] * len(env.reward_types())

    for type, value in state.typed_reward.items():
        d_reward[r_type[type]] = value

    return d_reward


def run_task(evaluation_config, network_config, reinforce_config):
    env = TowerExample()

    max_episode_steps = 10000

    state = env.reset()

    TOWER_BR, TOWER_TR, TOWER_BL, TOWER_TL = [1, 2, 3, 4]

    choose_tower = HRAAdaptive(name = "tower",
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
        tower_to_kill, q_values, _ = choose_tower.predict(state.state)
        end_time = time.time()

        action = env.new_action()

        env_start_time = time.time()

        action.attack_quadrant(tower_to_kill)
        state = env.act(action)
        env.reward_types()
        d_reward = decompose_reward(env, state)
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


    choose_tower.explanation = True

    explanation = Explanation("Tower Capture", (40,40))
    chart = BarChart("Move Explanation", "Actions", "QVal By Reward Type")
    layer_names = ["HP", "Type 1", "Type 2", "Type 3", "Friend", "Enemy"]


    #Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset(visualize=evaluation_config.render, record=True)
        total_reward = 0
        episode_summary = tf.Summary()

        tower_to_kill, q_values, saliencies = choose_tower.predict(state.state)
        choices = env.actions()['actions']

        for choice, action_value in choices.items():
            key = choice

            explanation.add_layers(layer_names, saliencies[action_value - 1][-1], key = key)
            group = BarGroup("Attack {}".format(choice), saliency_key = key)

            for index, r_type in enumerate(env.reward_types()):
                key = "{}Bar{}".format(choice, r_type)
                print(key, q_values[index][action_value-1])
                bar = Bar(r_type, q_values[index][action_value-1], saliency_key = key)
                explanation.add_layers(layer_names, saliencies[action_value-1][index], key=key)
                group.add_bar(bar)

            chart.add_bar_group(group)

        explanation.with_bar_chart(chart)


        action = env.new_action()
        action.attack_quadrant(tower_to_kill)
        action.skip = False if  evaluation_config.render else True

        state = env.act(action, explanation=explanation)

        print(env.reward_types())
        print("TYPED REWARD!!")
        print(q_values, state.typed_reward, state.reward)
        print("*****************")


        while not state.is_terminal():
            time.sleep(0.5)
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
