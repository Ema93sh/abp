import gym
import numpy as np
import abp

from abp import DQAdaptive


def run_task(config):
    config.name = "TicTacToe-v0"

    env_spec = gym.make(config.name)
    state = env_spec.reset()
    max_episode_steps = env_spec._max_episode_steps

    config.size_rewards = 9
    config.size_features = len(state)
    config.action_size = env_spec.action_space.n

    agent = DQAdaptive(config)

    #Episodes for training
    for epoch in range(config.training_episode):
        state = env_spec.reset()
        for steps in range(max_episode_steps):

            action = agent.predict(state)
            state, reward, done, info = env_spec.step(action)

            reshaped_board = np.reshape(info['board'], (3,3))

            sum_rows = np.sum(reshaped_board, axis = 1)
            sum_cols = np.sum(reshaped_board, axis = 0)
            sum_diagonal = np.trace(reshaped_board)
            sum_rev_diagonal = np.trace(np.flipud(reshaped_board))

            reward_type = 0

            for row in range(3):
                if sum_rows[row] == 3:
                    agent.reward(reward_type + row, 10)
                elif sum_rows[row] == -3:
                    agent.reward(reward_type + row, -10)

            reward_type = 3

            for col in range(3):
                if sum_cols[col] == 3:
                    agent.reward(reward_type + col, 10)
                elif sum_cols[col] == -3:
                    agent.reward(reward_type + col, -10)

            if sum_diagonal == 3:
                agent.reward(6, 10)
            elif sum_diagonal == -3:
                agent.reward(6, -10)

            if sum_rev_diagonal == 3:
                agent.reward(7, 10)
            elif sum_diagonal == -3:
                agent.reward(7, -10)

            if info['illegal_move']:
                agent.reward(8, -10)
            else:
                agent.reward(8, 1)

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
                    print("Ended cause of illegal move")
                if info['x_won']:
                    print("You WIN")
                elif info['o_won']:
                    print("You LOST")
                else:
                    print("DRAW")

                print("END OF EPISODE")
                agent.end_episode(state)
                break

    env_spec.close()
