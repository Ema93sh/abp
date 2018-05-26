import sys
import time

import numpy as np

from absl import app
from absl import flags

from pysc2.env import sc2_env, environment
from pysc2.lib import actions


from abp import HRAAdaptive
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter


from .utils.action import ActionWrapper
from .utils.reward import RewardWrapper


def run_task(evaluation_config, network_config, reinforce_config):
    flags.DEFINE_bool("use_feature_units", True,
                  "Whether to include feature units.")

    flags.FLAGS(sys.argv[:1]) #TODO Fix this!

    env = sc2_env.SC2Env( map_name = "CollectMineralShards",
                          step_mul = 10,
                          visualize = False,
                          save_replay_episodes = 0,
                          replay_dir = 'replay',
                          game_steps_per_episode = 10000,
                          use_feature_units = True,
                          feature_screen_size = 10,
                          feature_minimap_size = 10)
    choices =  ["Up", "Down", "Left", "Right"]

    reward_types = [(x, y)  for x in range(10) for y in range(10)]


    # Configure network for reward type
    networks = []
    for reward_type in reward_types:
        name = reward_type
        layers = [250]
        networks.append({"name": name, "layers": layers})

    network_config.networks = networks

    agent = HRAAdaptive(name = "ShardsCollector",
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
        actions = ActionWrapper(state).select(["SelectMarine1"])
        reward_wrapper = RewardWrapper(state, reward_types)
        state = env.step(actions)
        total_reward = 0
        done = False
        steps = 0
        while steps < 1000 and not done:
            steps += 1
            action, q_values = agent.predict(state[0].observation.feature_screen)

            actions = ActionWrapper(state).select([action])

            state = env.step(actions)

            decomposed_reward = reward_wrapper.reward(state)

            for reward_type in reward_types:
                agent.reward(reward_type, decomposed_reward[reward_type])

            total_reward += sum(decomposed_reward.values())
            done = state[0].step_type == environment.StepType.LAST


        agent.end_episode(state[0].observation.feature_screen)

        test_summary_writer.add_scalar(tag="Train/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)

        train_summary_writer.add_scalar(tag="Train/Steps to collect all shards", scalar_value=steps + 1,
                                        global_step=episode + 1)

    agent.disable_learning()

    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while steps < 1000 and not done:
            steps += 1
            action, q_values = agent.predict(state[0].observation.feature_screen)

            actions = ActionWrapper(state).select([action])

            state = env.step(actions)

            total_reward += sum([obs.reward for obs in state])
            done = state[0].step_type == environment.StepType.LAST

        test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)
        test_summary_writer.add_scalar(tag="Test/Steps to collect all Fruits", scalar_value=steps + 1,
                                       global_step=episode + 1)



    env.close()
