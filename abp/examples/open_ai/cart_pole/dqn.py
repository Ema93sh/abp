import gym
from abp import DQNAdaptive
from abp.utils import clear_summary_path
from tensorboardX import SummaryWriter


def run_task(evaluation_config, network_config, reinforce_config, log=True):
    env = gym.make(evaluation_config.env)
    max_episode_steps = env._max_episode_steps
    state = env.reset()

    threshold_angle = 0.087266463
    threshold_x = 1.5
    LEFT, RIGHT = [0, 1]

    agent = DQNAdaptive(name="cartpole",
                        choices=[LEFT, RIGHT],
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
        for steps in range(max_episode_steps):
            action, q_values = agent.predict(state)
            state, reward, done, info = env.step(action)
            cart_position, cart_velocity, pole_angle, pole_velocity = state

            agent.reward(reward)  # Reward for every step

            # Reward for pole angle increase or decrease
            if -threshold_angle < pole_angle < threshold_angle:
                agent.reward(1)
            else:
                agent.reward(-1)

            if steps < max_episode_steps and done:
                agent.reward(-40)  # Reward for terminal state

            if -threshold_x < cart_position < threshold_x:
                agent.reward(1)
            else:
                agent.reward(-1)

            total_reward += reward

            if done:
                agent.end_episode(state)
                if log:
                    train_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                    global_step=episode + 1)
                break

    # train_summary_writer.flush()

    agent.disable_learning()

    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_episode_steps):
            if evaluation_config.render:
                env.render()

            action, q_values = agent.predict(state)

            state, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                if log:
                    test_summary_writer.add_scalar(tag="Episode Reward", scalar_value=total_reward,
                                                   global_step=episode + 1)
                    print('Episode Reward:', total_reward)
                break

    env.close()
    pass
