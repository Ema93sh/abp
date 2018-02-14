import sys

import gym
# import tensorflow as tf

from abp import HRAAdaptive
from abp.utils import clear_summary_path

# TODO chart
from abp.utils.histogram import MultiQHistogram


def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = env._max_episode_steps
    state = env.reset()

    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]

    HOME_TYPE, TREASURE_TYPE, TERRAIN_TYPE = [0, 1, 2]

    traveller = HRAAdaptive(name="traveller",
                            choices=[LEFT, RIGHT, UP, DOWN],
                            network_config=network_config,
                            reinforce_config=reinforce_config)

    # training_summaries_path = evaluation_config.summaries_path + "/train"
    # clear_summary_path(training_summaries_path)
    # train_summary_writer = tf.summary.FileWriter(training_summaries_path)
    #
    #
    # test_summaries_path = evaluation_config.summaries_path + "/test"
    # clear_summary_path(test_summaries_path)
    # test_summary_writer = tf.summary.FileWriter(test_summaries_path)

    # Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        # episode_summary = tf.Summary()
        for steps in range(max_episode_steps):
            action, _ = traveller.predict(state)
            state, reward, done, info = env.step(action)
            total_reward += reward

            reward = info["decomposed_reward"]

            # TODO check if reward.values() always gives the same order
            traveller.reward(list(reward.values()))

            # TODO add option for type annotation instead of list of rewards
            # traveller.reward(HOME_TYPE, reward["HOME"])
            # traveller.reward(TREASURE_TYPE, reward["TREASURE"])
            # traveller.reward(TERRAIN_TYPE, reward["TERRAIN"])

            if done:
                traveller.end_episode(state)
                # episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                # train_summary_writer.add_summary(episode_summary, episode + 1)
                print(episode + 1, 'Episode Reward:', total_reward)
                break

    traveller.disable_learning()

    # TODO chart
    chart = MultiQHistogram(traveller.reward_types, len(traveller.choices), ("Left", "Right", "Up", "Down"), ylim=5)
    q_labels = ["Home", "Treasure", "Terrain"]
    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        action = None
        state = env.reset()
        total_reward = 0
        # episode_summary = tf.Summary()

        for steps in range(max_episode_steps):
            action, q_values = traveller.predict(state)

            if evaluation_config.render:
                s = env.render(mode="ansi")
                print(s.getvalue())
                print("Press enter to continue:")
                sys.stdin.read(1)
                chart.render(q_values, q_labels)
                # import pdb; pdb.set_trace()

            state, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                if evaluation_config.render:
                    s = env.render(mode="ansi")
                    print(s.getvalue())
                    print("********** END OF EPISODE *********")
                # episode_summary.value.add(tag="Episode Reward", simple_value=total_reward)
                # test_summary_writer.add_summary(episode_summary, episode + 1)
                print(episode + 1, 'Episode Reward:', total_reward)
                break

    env.close()
