import sys
import time

from absl import flags

from pysc2.env import sc2_env, environment


from abp import HRAAdaptive
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter


from .utils.action import ActionWrapper
from .utils.reward import RewardWrapper


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
                         feature_screen_size=10,
                         feature_minimap_size=10)

    choices = ["Up", "Down", "Left", "Right"]

    pdx_explanation = PDX()

    reward_types = [(x, y) for x in range(10) for y in range(10)]
    reward_names = ["loc (%d, %d)" % (x, y) for x, y in reward_types]

    # Configure network for reward type
    networks = []
    for reward_type in reward_types:
        name = reward_type
        layers = [250]
        networks.append({"name": name, "layers": layers})

    network_config.networks = networks

    agent = HRAAdaptive(name="ShardsCollector",
                        choices=choices,
                        reward_types=reward_types,
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
        actions = ActionWrapper(state).select(["SelectMarine1"])
        reward_wrapper = RewardWrapper(state, reward_types)
        state = env.step(actions)
        total_reward = 0
        done = False
        steps = 0
        model_time = 0
        env_time = 0
        while not done:
            steps += 1
            model_start_time = time.time()
            action, q_values, combined_q_values = agent.predict(
                state[0].observation.feature_screen.player_relative.flatten())

            model_time += (time.time() - model_start_time)

            actions = ActionWrapper(state).select([action])

            env_time -= time.time()
            state = env.step(actions)
            env_time += time.time()

            decomposed_reward = reward_wrapper.reward(state)

            for reward_type in reward_types:
                agent.reward(reward_type, decomposed_reward[reward_type])

            total_reward += sum(decomposed_reward.values())
            done = state[0].step_type == environment.StepType.LAST

        agent.end_episode(state[0].observation.feature_screen.player_relative.flatten())

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
        actions = ActionWrapper(state).select(["SelectMarine1"])
        reward_wrapper = RewardWrapper(state, reward_types)
        state = env.step(actions)
        total_reward = 0
        done = False
        steps = 0
        model_time = 0
        while steps < 1000 and not done:
            steps += 1
            model_start_time = time.time()
            action, q_values, combined_q_values = agent.predict(
                state[0].observation.feature_screen.player_relative.flatten())

            if evaluation_config.render:
                action_index = choices.index(action)
                combined_q_values = combined_q_values.data.numpy()
                q_values = q_values.data.numpy()
                pdx_explanation.render_decomposed_rewards(
                    action_index, combined_q_values, q_values, choices, reward_names)
                pdx_explanation.render_all_pdx(action_index, len(
                    choices), q_values, choices, reward_names)
                time.sleep(1)

            model_time += (time.time() - model_start_time)

            actions = ActionWrapper(state).select([action])

            state = env.step(actions)

            decomposed_reward = reward_wrapper.reward(state)

            total_reward += sum(decomposed_reward.values())
            done = state[0].step_type == environment.StepType.LAST

        print("Episode", episode + 1, total_reward)

        test_summary_writer.add_scalar(tag="Test/Episode Reward",
                                       scalar_value=total_reward,
                                       global_step=episode + 1)
        test_summary_writer.add_scalar(tag="Test/Steps to collect all Fruits",
                                       scalar_value=steps + 1,
                                       global_step=episode + 1)

    env.close()
