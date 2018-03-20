import gym
import tensorflow as tf

from abp import A2CAdaptive
from abp.utils import clear_summary_path

def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = env._max_episode_steps
    state = env.reset()

    threshold_angle = 0.087266463
    threshold_x = 1.5
    LEFT, RIGHT = [0, 1]

    agent = A2CAdaptive(name = "cartpole",
                        choices = [LEFT, RIGHT],
                        network_config = network_config,
                        reinforce_config = reinforce_config)

    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = tf.summary.FileWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = tf.summary.FileWriter(test_summaries_path)


    #Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()
        for steps in range(max_episode_steps):
            action, actor_prob, critic_values = agent.predict(state)
            state, reward, done, info = env.step(action)
            cart_position, cart_velocity, pole_angle, pole_velocity = state

            agent.reward(reward) # Reward for every step

            total_reward += reward

            if done:
                agent.end_episode(state)
                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                train_summary_writer.add_summary(episode_summary, episode + 1)
                break

    train_summary_writer.flush()

    agent.disable_learning()


    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        episode_summary = tf.Summary()
        for step in range(max_episode_steps):
            if evaluation_config.render:
                env.render()

            action, actor_prob, critic_values = agent.predict(state)

            state, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                test_summary_writer.add_summary(episode_summary, episode + 1)
                break

    test_summary_writer.flush()

    env.close()
    pass
