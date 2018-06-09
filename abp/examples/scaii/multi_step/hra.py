import operator
import logging

import time

from scaii.env.sky_rts.env.scenarios.tower_example import TowerExample
from scaii.env.explanation import Explanation as SkyExplanation, BarChart, BarGroup, Bar

from abp import HRAAdaptive
from abp.explanations import Saliency

logger = logging.getLogger('root')


def run_task(evaluation_config, network_config, reinforce_config):
    env = TowerExample("multi_step")

    reward_types = sorted(env.reward_types())
    decomposed_rewards = {}

    for type in reward_types:
        decomposed_rewards[type] = 0

    state = env.reset()

    actions = env.actions()['actions']
    actions = sorted(actions.items(), key=operator.itemgetter(1))
    choice_descriptions = list(map(lambda x: x[0], actions))
    choices = list(map(lambda x: x[1], actions))

    # Configure network for reward type
    networks = []
    for reward_type in reward_types:
        name = reward_type
        layers = [{"type": "FC", "neurons": 50}]
        networks.append({"name": name, "layers": layers})

    network_config.networks = networks

    choose_tower = HRAAdaptive(name="tower",
                               choices=choices,
                               reward_types=reward_types,
                               network_config=network_config,
                               reinforce_config=reinforce_config)

    # Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        step = 1

        while not state.is_terminal():
            step += 1
            (tower_to_kill,
             q_values,
             combined_q_values) = choose_tower.predict(state.state.flatten())

            action = env.new_action()
            action.attack_quadrant(tower_to_kill)
            action.skip = True
            state = env.act(action)

            for reward_type, reward in state.typed_reward.items():
                choose_tower.reward(reward_type, reward)
                total_reward += reward

        choose_tower.end_episode(state.state.flatten())

        logger.debug("Episode %d : %d, Step: %d" % (episode + 1, total_reward, step))

    choose_tower.disable_learning()

    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        layer_names = ["HP", "Agent Location", "Small Towers", "Big Towers", "Friend", "Enemy"]

        saliency_explanation = Saliency(choose_tower)

        state = env.reset(visualize=evaluation_config.render, record=True)
        total_reward = 0
        step = 0

        while not state.is_terminal():
            step += 1
            explanation = SkyExplanation("Tower Capture", (40, 40))
            (tower_to_kill,
             q_values,
             combined_q_values) = choose_tower.predict(state.state.flatten())

            q_values = q_values.data.numpy()
            combined_q_values = combined_q_values.data.numpy()
            saliency_explanation.generate_saliencies(
                step, state.state.flatten(),
                choice_descriptions,
                layer_names,
                reshape=state.state.shape)

            decomposed_q_chart = BarChart("Q Values", "Actions", "QVal By Reward Type")
            for choice_idx, choice in enumerate(choices):
                key = choice_descriptions[choice_idx]
                group = BarGroup("Attack {}".format(key))

                for reward_index, reward_type in enumerate(reward_types):
                    key = "{}_{}".format(choice, reward_type)
                    bar = Bar(reward_type, q_values[reward_index][choice_idx])
                    group.add_bar(bar)

                decomposed_q_chart.add_bar_group(group)

            explanation.with_bar_chart(decomposed_q_chart)

            action = env.new_action()
            action.attack_quadrant(tower_to_kill)
            action.skip = True

            state = env.act(action, explanation=explanation)

            time.sleep(0.5)

            total_reward += state.reward

        logger.info("End Episode of episode %d with %d steps" % (episode + 1, step))
        logger.info("Total Reward %d!" % (total_reward))
