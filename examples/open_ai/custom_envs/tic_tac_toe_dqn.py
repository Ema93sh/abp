import gym
import numpy as np
import abp.envs

from abp.adaptives.dqn import DQNAdaptive


env_spec = gym.make("TicTacToe-v0")
max_episode_steps = env_spec._max_episode_steps

training_episode = 80 * 1000
test_episodes = 100

state = env_spec.reset()

agent = DQNAdaptive(env_spec.action_space.n, len(state), "Tic Tac Toe", decay_steps = 2000)

for epoch in range(training_episode):
    state = env_spec.reset()
    for steps in range(max_episode_steps):
        action = agent.predict(state)
        state, reward, done, info = env_spec.step(action)

        reshaped_board = np.reshape(state, (3,3))

        sum_rows = np.sum(reshaped_board, axis = 1)
        sum_cols = np.sum(reshaped_board, axis = 0)
        sum_diagonal = np.trace(reshaped_board)
        sum_rev_diagonal = np.trace(np.flipud(reshaped_board))

        reward_type = 0

        for row in range(3):
            if sum_rows[row] == 3:
                agent.reward(10)
            elif sum_rows[row] == -3:
                agent.reward(-10)

        reward_type = 3

        for col in range(3):
            if sum_cols[col] == 3:
                agent.reward(10)
            elif sum_cols[col] == -3:
                agent.reward(-10)

        if sum_diagonal == 3:
            agent.reward(10)
        elif sum_diagonal == -3:
            agent.reward(-10)

        if sum_rev_diagonal == 3:
            agent.reward(10)
        elif sum_diagonal == -3:
            agent.reward(-10)

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
for epoch in range(test_episodes):
    state = env_spec.reset()
    for steps in range(max_episode_steps):
        env_spec.render()
        action = agent.predict(state)
        state, reward, done, info = env_spec.step(action)
        agent.test_reward(reward)

        if done:
            env_spec.render()
            if info['illegal_move']:
                print "Ended cause of illegal move", action
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
