import gym
from gym import wrappers
from abp.dqn.adaptive import Adaptive

env_spec = gym.make("CartPole-v0")

threshold_angle = 0.087266463
threshold_x = 1.5

max_episode_steps = env_spec._max_episode_steps

state = env_spec.reset()

agent = Adaptive(env_spec.action_space.n, len(state), "Cart Pole", decay_steps = 250)

training_episode = 300

test_episodes = 100

#Episodes
for epoch in range(training_episode):
    state = env_spec.reset()
    for steps in range(max_episode_steps):
        action = agent.predict(state)
        state, reward, done, info = env_spec.step(action)
        cart_position, cart_velocity, pole_angle, pole_velocity = state
        agent.reward(reward) # Reward for every step

        # Reward for pole angle increase or decrease
        if  -threshold_angle < pole_angle < threshold_angle:
            agent.reward(1)
        else:
            agent.reward(-1)

        if steps < max_episode_steps and done:
            agent.reward(-40) # Reward for terminal state

        if -threshold_x < cart_position < threshold_x:
            agent.reward(1)
        else:
            agent.reward(-1)

        agent.actual_reward(reward)

        if done:
            agent.end_episode(state)
            break


agent.disable_learning()


# After learning Episodes
for epoch in range(test_episodes):
    state = env_spec.reset()
    for t in range(max_episode_steps):
        env_spec.render()
        action = agent.predict(state)
        state, reward, done, info = env_spec.step(action)
        agent.test_reward(reward)

        if done:
            agent.end_episode(state)
            break

env_spec.close()
