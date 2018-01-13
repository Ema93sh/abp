import time

import gym
import numpy as np
import tensorflow as tf

from abp import DQNAdaptive
from abp.utils import clear_summary_path


def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = env._max_episode_steps
    state = env.reset()
    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]

    agent = DQNAdaptive(name="traveller",
                        choices = [LEFT, RIGHT, UP, DOWN],
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
            action, _ = agent.predict(state)
            state, reward, done, info = env.step(action)
            total_reward += reward

            traveller_location = info['traveller_location']

            if np.where(info['mountain_locations'] == traveller_location):
                agent.reward(-4)
            if np.where(info['river_locations'] == traveller_location):
                agent.reward(-2)
            if np.where(info['hill_locations'] == traveller_location):
                agent.reward(-2)

            if np.where(info['gold_locations'] == traveller_location):
                agent.reward(2)
            if np.where(info['diamond_locations'] == traveller_location):
                agent.reward(3)

            if done:
                if traveller_location == info['house_location']:
                    agent.reward(10)
                    total_reward += reward
                else:
                    agent.reward(-10)
                    total_reward += reward

                agent.end_episode(state)
                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                train_summary_writer.add_summary(episode_summary, episode + 1)
                break


    agent.disable_learning()

    #TODO disable render based on the evaluation_config

    #Test Episodes
    chart = SingleQBarChart(env.action_space.n, ('Left', 'Right', 'Up', 'Down'),  y_lim = 20)

    import curses

    screen = curses.initscr()
    curses.savetty()
    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)
    # screen.nodelay(0)
    screen.keypad(1)


    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        reward = 0
        action = None
        state = env.reset()
        days_remaining = 8
        total_reward = 0
        episode_summary = tf.Summary()
        for steps in range(max_episode_steps):
            screen.clear()
            s = env.render(mode='ansi')
            screen.addstr(s.getvalue())
            screen.addstr("Days Remaining: " + str(days_remaining) + "\n")
            screen.refresh()

            action, q_values = agent.predict(state)
            chart.render(q_values)
            time.sleep(1)

            state, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                episode_summary.value.add(tag = "Episode Reward", simple_value = total_reward)
                test_summary_writer.add_summary(episode_summary, episode + 1)
                agent.end_episode(state)
                screen.clear()
                s = env.render(mode='ansi')
                screen.addstr(s.getvalue())
                screen.addstr("End Episode\n")
                screen.refresh()
                time.sleep(1)
                break

    env.close()
