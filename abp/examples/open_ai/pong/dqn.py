import gym
import numpy as np
from abp import DQNAdaptive
from tensorboardX import SummaryWriter


def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = env._max_episode_steps
    state = env.reset()

    UP, DOWN = [2, 3]
    choices = [UP, DOWN]

    agent = DQNAdaptive(name="Pong",
                        choices=choices,
                        network_config=network_config,
                        reinforce_config=reinforce_config)

    training_summaries_path = evaluation_config.summaries_path + "/train"
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    test_summary_writer = SummaryWriter(test_summaries_path)

    # Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        for steps in range(max_episode_steps):
            action, q_values = agent.predict(np.rollaxis(state, 2))

            state, reward, done, info = env.step(action)

            agent.reward(reward)  # Reward for every step

            total_reward += reward

            if done:
                agent.end_episode(np.rollaxis(state, 2))
                train_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                    global_step=episode + 1)
                break

    agent.disable_learning()

    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_episode_steps):
            if evaluation_config.render:
                env.render()

            action, q_values = agent.predict(np.rollaxis(state, 2))

            state, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                test_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                               global_step=episode + 1)
                break

    env.close()
