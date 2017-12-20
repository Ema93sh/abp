import gym
import numpy as np
import abp
from abp import QAdaptive

def run_task(config):
    config.name = "TicTacToe-v0"

    env_spec = gym.make(config.name)
    state = env_spec.reset()
    max_episode_steps = env_spec._max_episode_steps


    config.size_features = len(state)
    config.action_size = env_spec.action_space.n

    agent = QAdaptive(config)

    for epoch in range(config.training_episode):
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            action = agent.predict(state)

            state, reward, done, info = env_spec.step(action)

            if done:
                if info['x_won'] == True:
                    agent.reward(10)
                elif info['o_won'] == True:
                    agent.reward(-10)
                else:
                    agent.reward(5)


            if info['illegal_move']:
                agent.reward(-10)
            else:
                agent.reward(1)

            agent.actual_reward(reward)

            if done:
                agent.end_episode(state)
                break

    agent.disable_learning()

    # After learning Episodes
    for epoch in range(config.test_episodes):
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            if config.render:
                env_spec.render()
            action = agent.predict(state)
            state, reward, done, info = env_spec.step(action)
            agent.test_reward(reward)

            if done:
                env_spec.render()
                if info['illegal_move']:
                    print "Ended cause of illegal move"
                if info['x_won']:
                    print "You WIN"
                elif info['o_won']:
                    print "You LOST"
                else:
                    print "DRAW"

                print "END OF EPISODE"
                agent.end_episode(state)
                break

    env_spec.close()
