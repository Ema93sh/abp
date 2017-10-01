import gym
import numpy as np
import abp.custom_envs
from abp.adaptives.dqn import DQNAdaptive

def run_task(job_dir, render = True, training_episode = 80000, test_episodes = 100, decay_steps = 2000, model_path = None, restore_model = False):
    env_spec = gym.make("TicTacToe-v0")
    max_episode_steps = env_spec._max_episode_steps

    state = env_spec.reset()

    agent = DQNAdaptive(env_spec.action_space.n,
                        len(state),
                        "Tic Tac Toe",
                        job_dir = job_dir,
                        decay_steps = decay_steps,
                        model_path = model_path,
                        restore_model = restore_model)

    for epoch in range(training_episode):
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
    for epoch in range(test_episodes):
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            if render:
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
