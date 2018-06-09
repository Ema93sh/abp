import gym


if __name__ == '__main__':
    """ User interaction with the Environment"""
    env = gym.make("FruitCollection-v0")
    for ep in range(5):
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            env.render()
            print(obs)
            action = int(input("action:"))
            obs, rewards, done, info = env.step(action, decompose_reward=True)
            print(rewards)
            total_reward += sum(rewards.values())
        env.close()
        print("Episode Reward:", total_reward)
