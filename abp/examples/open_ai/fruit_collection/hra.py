import gym
import time
import numpy as np

from abp import HRAAdaptive
from abp.utils import clear_summary_path
from tensorboardX import SummaryWriter


def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = 300
    state = env.reset()
    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]
    choices = [LEFT, RIGHT, UP, DOWN]

    reward_types = env.reward_types

    agent = HRAAdaptive(name = "FruitCollecter",
                        choices = choices,
                        reward_types = reward_types,
                        network_config = network_config,
                        reinforce_config = reinforce_config)


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
        done = False
        steps = 0
        while not done:
            steps += 1
            action, q_values = agent.predict(state)
            state, rewards, done, info = env.step(action, decompose_reward = True)

            for reward_type in rewards.keys():
                agent.reward(reward_type, rewards[reward_type])

            total_reward += sum(rewards.values())

        agent.end_episode(state)
        test_summary_writer.add_scalar(tag="Train/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)
        train_summary_writer.add_scalar(tag="Train/Steps to collect all Fruits", scalar_value=steps + 1,
                                        global_step=episode + 1)

    agent.disable_learning()

    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            steps += 1
            action, q_values = agent.predict(state)
            if evaluation_config.render:
                env.render()
                time.sleep(0.5)

            state, reward, done, info = env.step(action)

            total_reward += sum(reward)

        agent.end_episode(state)

        test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)
        test_summary_writer.add_scalar(tag="Test/Steps to collect all Fruits", scalar_value=steps + 1,
                                       global_step=episode + 1)

    env.close()
