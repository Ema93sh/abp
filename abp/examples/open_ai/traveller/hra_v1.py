import sys

import gym
# import tensorflow as tf

from abp import HRAAdaptive
from abp.utils import clear_summary_path

# TODO chart
from abp.utils.histogram import MultiQHistogram
from tensorboardX import SummaryWriter

def run_task(evaluation_config, network_config, reinforce_config,log=True):
    env = gym.make(evaluation_config.env)
    max_episode_steps = 1000
    state = env.reset()

    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]
    LEFT, RIGHT, UP, DOWN, NOOP = [0, 1, 2, 3, 4]

    HOME_TYPE, TREASURE_TYPE, TERRAIN_TYPE = [0, 1, 2]

    traveller = HRAAdaptive(name="traveller",
                            choices=[LEFT, RIGHT, UP, DOWN],
                            network_config=network_config,
                            reinforce_config=reinforce_config)

    if log:
        training_summaries_path = evaluation_config.summaries_path + "/train"
        clear_summary_path(training_summaries_path)
        train_summary_writer = SummaryWriter(training_summaries_path)

        test_summaries_path = evaluation_config.summaries_path + "/test"
        clear_summary_path(test_summaries_path)
        test_summary_writer = SummaryWriter(test_summaries_path)

    # Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        # episode_summary = tf.Summary()
        for steps in range(max_episode_steps):
            action, _ = traveller.predict(state)
            state, reward, done, info = env.step(action, decompose_level = 1)
            total_reward += sum(reward)

            #TODO check if reward.values() always gives the same order
            traveller.reward(reward)

            # TODO add option for type annotation instead of list of rewards
            # traveller.reward(HOME_TYPE, reward["HOME"])
            # traveller.reward(TREASURE_TYPE, reward["TREASURE"])
            # traveller.reward(TERRAIN_TYPE, reward["TERRAIN"])

            if done:
                traveller.end_episode(state)
                if log:
                    train_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                    global_step=episode + 1)
                    print(episode + 1, 'Episode Reward:', total_reward)
                break

    traveller.disable_learning()

    # TODO chart
    chart = MultiQHistogram(traveller.reward_types, len(traveller.choices), ("Left", "Right", "Up", "Down"), ylim = 5)
    q_labels = ["Terrain", "Treasure", "Home", "Death"]
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
                if log:
                    test_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                    global_step=episode + 1)
                    print(episode + 1, 'Episode Reward:', total_reward)
                break

    env.close()
