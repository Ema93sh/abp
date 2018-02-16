import gym
from gym import wrappers
from abp import HRAAdaptive


def run_task(evaluation_config, network_config, reinforce_config):
    env = gym.make(evaluation_config.env)
    max_episode_steps = env._max_episode_steps
    state = env.reset()

    threshold_angle = 0.087266463
    threshold_x = 1.5

    LEFT, RIGHT = [0, 1]

    agent = HRAAdaptive(name="cartpole",
                        choices=[LEFT, RIGHT],
                        network_config=network_config,
                        reinforce_config=reinforce_config)

    # Episodes
    for epoch in range(evaluation_config.training_episodes):
        state = env.reset()
        for steps in range(max_episode_steps):
            action, q_values = agent.predict(state)
            state, reward, done, info = env.step(action)
            cart_position, cart_velocity, pole_angle, pole_velocity = state

            d_reward = [0] * 3

            # Reward for pole angle increase or decrease
            if -threshold_angle < pole_angle < threshold_angle:
                d_reward[0] = 1
            else:
                d_reward[0] = -1

            if steps < max_episode_steps and done:
                d_reward[1] = -40

            if -threshold_x < cart_position < threshold_x:
                d_reward[2] = 1
            else:
                d_reward[2] = -1

            agent.reward(d_reward)

            if done:
                agent.end_episode(state)
                break

    agent.disable_learning()

    # After learning Episodes
    for epoch in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0

        for steps in range(max_episode_steps):
            if evaluation_config.render:
                env.render()

            action, q_values = agent.predict(state)

            state, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                break

    env.close()
