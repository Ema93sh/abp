import gym

from baselines import deepq


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    # import pdb; pdb.set_trace()
    # is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 5
    # print(lcl['t'])
    return False


def main():
    env = gym.make("FruitCollection-v0")
    model = deepq.models.mlp([120])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=300 * 200,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=1,
        callback=callback
    )
    print("Saving model")
    act.save("fruitcollection_model.pkl")


if __name__ == '__main__':
    main()
