import gym
import time
import numpy as np
# import tensorflow as tf

from abp import HRAAdaptive
from abp.utils import clear_summary_path
from abp.openai.wrappers import RewardWrapper
from tensorboardX import SummaryWriter


# TODO
# *reward wrapper

def run_task(evaluation_config, network_config, reinforce_config, log=True):
    env = gym.make(evaluation_config.env)
    max_episode_steps = 300
    state = env.reset()
    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]

    agent = HRAAdaptive(name="FruitCollecter",
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
            action, q_values = agent.predict(state)
            state, reward, done, info = env.step(action, decompose_reward = True)

            agent.reward(reward)

            total_reward += sum(reward)

            if done or steps == (max_episode_steps - 1):
                agent.end_episode(state)
                if log:
                    train_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                    global_step=episode + 1)
                    train_summary_writer.add_scalar(tag="Steps to collect all Fruits", scalar_value=steps + 1,
                                                    global_step=episode + 1)
                    print('Episode Reward:', total_reward)
                    print('Steps to collect all fruits:', steps + 1)
                break

    # train_summary_writer.flush()

    agent.disable_learning()

    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        # episode_summary = tf.Summary()

        for steps in range(max_episode_steps):
            action, q_values = agent.predict(state)
            if evaluation_config.render:
                s = env.render(mode='ansi')
                print(s.getvalue())
                time.sleep(0.5)

            state, reward, done, info = env.step(action)

            total_reward += sum(reward)

            if done:
                agent.end_episode(state)
                if log:
                    test_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                   global_step=episode + 1)
                    test_summary_writer.add_scalar(tag="Steps to collect all Fruits", scalar_value=steps + 1,
                                                   global_step=episode + 1)
                    print('Episode Reward:', total_reward)
                    print('Steps to collect all fruits:', steps + 1)

                break

    # test_summary_writer.flush()
    env.close()
