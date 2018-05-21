import sys

import gym

from abp import HRAAdaptive
from abp.utils import clear_summary_path

from abp.utils.histogram import MultiQHistogram
from tensorboardX import SummaryWriter

def run_task(evaluation_config, network_config, reinforce_config, log=True):
    env = gym.make(evaluation_config.env)
    max_episode_steps = 1000
    state = env.reset()

    network_config.input_shape = [len(state)]

    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]

    reward_types = sorted(["Home", "Treasure", "Terrain", "Death"])

    traveller = HRAAdaptive(name="traveller",
                            choices=[LEFT, RIGHT, UP, DOWN],
                            reward_types = reward_types,
                            network_config=network_config,
                            reinforce_config=reinforce_config)

    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)

    # Training Episodes
    goal = 0
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        for steps in range(max_episode_steps):
            action, _ = traveller.predict(state)

            state, reward, done, info = env.step(action, decompose_level = 1)

            total_reward += sum(reward.values())


            traveller.reward("Home", reward["HOME"])
            traveller.reward("Treasure", reward["TREASURE"])
            traveller.reward("Terrain", reward["TERRAIN"])
            traveller.reward("Death", reward["DEATH"])

            if done:
                if total_reward >= 20:
                    goal += 1
                    print("*****************************************************************")

                traveller.end_episode(state)
                train_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                    global_step=episode + 1)
                print(episode + 1, 'Episode Reward:', total_reward, "Goal:", goal)
                break

    traveller.disable_learning()

    # TODO chart
    chart = MultiQHistogram(len(traveller.reward_types), len(traveller.choices), ("Left", "Right", "Up", "Down"), ylim = 5)
    q_labels = ["Home", "Treasure", "Terrain", "Death"]
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
                print("LEFT", "RIGHT", "UP", "DOWN")
                for q_label, q_value in zip(reward_types, q_values):
                    print(q_label, q_value)
                print("Press enter to continue:")
                chart.render(q_values, reward_types)
                sys.stdin.read(1)


            state, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                if evaluation_config.render:
                    s = env.render(mode="ansi")
                    print(s.getvalue())
                    print("********** END OF EPISODE *********")
                    test_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                    global_step=episode + 1)
                    print(episode + 1, 'Episode Reward:', total_reward)
                break

    env.close()
