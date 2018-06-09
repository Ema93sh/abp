import sys
import time

from absl import flags
from pysc2.env import sc2_env, environment


from abp import DQNAdaptive
from abp.utils import clear_summary_path
from tensorboardX import SummaryWriter
from .utils.action import ActionWrapper


def run_task(evaluation_config, network_config, reinforce_config):
    flags.DEFINE_bool("use_feature_units", True,
                      "Whether to include feature units.")

    flags.FLAGS(sys.argv[:1])  # TODO Fix this!

    env = sc2_env.SC2Env(map_name="CollectMineralShards",
                         step_mul=8,
                         visualize=False,
                         save_replay_episodes=0,
                         replay_dir='replay',
                         game_steps_per_episode=10000,
                         use_feature_units=True,
                         feature_screen_size=32,
                         feature_minimap_size=32)

    choices = ["Up", "Down", "Left", "Right"]

    agent = DQNAdaptive(name="ShardsCollector",
                        choices=choices,
                        network_config=network_config,
                        reinforce_config=reinforce_config)

    training_summaries_path = evaluation_config.summaries_path + "/train"

    if evaluation_config.training_episodes > 0:
        clear_summary_path(training_summaries_path)

    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)

    # Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        actions = ActionWrapper(state, grid_size=32).select(["SelectMarine1"])
        state = env.step(actions)
        total_reward = 0
        done = False
        steps = 0
        model_time = 0
        while not done:
            steps += 1
            model_start_time = time.time()
            action, q_values = agent.predict(state[0].observation.feature_screen)
            model_time += (time.time() - model_start_time)

            actions = ActionWrapper(state, grid_size=32).select([action])

            state = env.step(actions)

            agent.reward(state[0].reward)

            total_reward += state[0].reward

            done = state[0].step_type == environment.StepType.LAST

        agent.end_episode(state[0].observation.feature_screen)

        test_summary_writer.add_scalar(tag="Train/Episode Reward",
                                       scalar_value=total_reward,
                                       global_step=episode + 1)

        train_summary_writer.add_scalar(tag="Train/Steps to collect all shards",
                                        scalar_value=steps + 1,
                                        global_step=episode + 1)

    agent.disable_learning()

    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        actions = ActionWrapper(state, grid_size=32).select(["SelectMarine1"])
        state = env.step(actions)
        total_reward = 0
        done = False
        steps = 0
        model_time = 0
        while steps < 1000 and not done:
            steps += 1
            model_start_time = time.time()
            action, q_values = agent.predict(state[0].observation.feature_screen)

            if evaluation_config.render:
                time.sleep(evaluation_config.sleep)

            model_time += (time.time() - model_start_time)

            actions = ActionWrapper(state, grid_size=32).select([action])

            state = env.step(actions)

            total_reward += state[0].reward

            done = state[0].step_type == environment.StepType.LAST

        test_summary_writer.add_scalar(tag="Test/Episode Reward",
                                       scalar_value=total_reward,
                                       global_step=episode + 1)
        test_summary_writer.add_scalar(tag="Test/Steps to collect all Fruits",
                                       scalar_value=steps + 1,
                                       global_step=episode + 1)

    env.close()
