import copy
import operator
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

def run_task(evaluation_config, network_config, reinforce_config):
    env = TowerExample()

    reward_types = sorted(env.reward_types())
    decomposed_rewards = {}

    for type in reward_types:
        decomposed_rewards[type] = 0

    max_episode_steps = 10000

    state = env.reset()

    actions = env.actions()['actions']
    actions = sorted(actions.items(), key=operator.itemgetter(1))
    choice_descriptions = list(map(lambda x: x[0], actions))
    choices = list(map(lambda x: x[1], actions))

    choose_tower = HRAAdaptive(name = "tower",
                               choices = choices,
                               reward_types = reward_types,
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
        tower_to_kill, q_values, _ = choose_tower.predict(state.state)

        action = env.new_action()

        action.attack_quadrant(tower_to_kill)
        state = env.act(action)

        for reward_type, reward in state.typed_reward.items():
            choose_tower.reward(reward_type, reward)

        total_reward += state.reward

        choose_tower.end_episode(state.state)

        logger.info("Episode %d : %d" % (episode + 1, total_reward))
        episode_summary.value.add(tag = "Reward", simple_value = total_reward)
        train_summary_writer.add_summary(episode_summary, episode + 1)

    train_summary_writer.flush()

    choose_tower.disable_learning()

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = tf.summary.FileWriter(test_summaries_path)


    # choose_tower.explanation = True

    explanation = Explanation("Tower Capture", (40,40))
    chart = BarChart("Move Explanation", "Actions", "QVal By Reward Type")
    layer_names = ["HP", "Type 1", "Type 2", "Type 3", "Friend", "Enemy"]


    choose_tower.enable_explanation()

    #Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset(visualize=evaluation_config.render, record=True)
        total_reward = 0
        episode_summary = tf.Summary()

        tower_to_kill, q_values, saliencies = choose_tower.predict(state.state)

        for choice_idx, choice in enumerate(choices):
            key = choice_descriptions[choice_idx]
            explanation.add_layers(layer_names, saliencies[choice]["all"], key = key)
            group = BarGroup("Attack {}".format(key), saliency_key = key)

            for reward_index, reward_type in enumerate(reward_types):
                key = "{}_{}".format(choice, reward_type)
                print(key, q_values[reward_index][choice_idx])
                bar = Bar(reward_type, q_values[reward_index][choice_idx], saliency_key = key)
                explanation.add_layers(layer_names, saliencies[choice][reward_type], key=key)
                group.add_bar(bar)

            chart.add_bar_group(group)

        explanation.with_bar_chart(chart)


        action = env.new_action()
        action.attack_quadrant(tower_to_kill)
        action.skip = False if  evaluation_config.render else True

        state = env.act(action, explanation = explanation)

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
