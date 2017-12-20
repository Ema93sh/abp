import gym
import numpy as np
import abp
import time

from abp import DQNAdaptive

from abp.utils.bar_chart import SingleQBarChart

def run_task(config):
    config.name = "Traveller-v0"

    env_spec = gym.make(config.name)
    max_episode_steps = env_spec._max_episode_steps
    state = env_spec.reset()

    config.size_features = len(state)
    config.action_size = env_spec.action_space.n

    agent = DQNAdaptive(config)

    #Training Episodes
    for epoch in range(config.training_episode):
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            action, _ = agent.predict(state)
            state, reward, done, info = env_spec.step(action)
            traveller_location = info['traveller_location']

            if np.where(info['mountain_locations'] == traveller_location):
                agent.reward(-4)
            if np.where(info['river_locations'] == traveller_location):
                agent.reward(-2)
            if np.where(info['hill_locations'] == traveller_location):
                agent.reward(-2)

            if np.where(info['gold_locations'] == traveller_location):
                agent.reward(2)
            if np.where(info['diamond_locations'] == traveller_location):
                agent.reward(3)

            agent.actual_reward(reward)

            if done:
                if traveller_location == info['house_location']:
                    agent.reward(10)
                else:
                    agent.reward(-10)

                agent.end_episode(state)
                break


    agent.disable_learning()

    #Test Episodes
    chart = SingleQBarChart(env_spec.action_space.n, ('Left', 'Right', 'Up', 'Down'),  y_lim = 20)

    import curses

    screen = curses.initscr()
    curses.savetty()
    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)
    # screen.nodelay(0)
    screen.keypad(1)


    # Test Episodes
    for epoch in range(config.test_episodes):
        reward = 0
        action = None
        state = env_spec.reset()
        days_remaining = 8
        for steps in range(max_episode_steps):
            screen.clear()
            s = env_spec.render(mode='ansi')
            screen.addstr(s.getvalue())
            screen.addstr("Days Remaining: " + str(days_remaining) + "\n")
            screen.refresh()

            action, q_values = agent.predict(state)
            chart.render(q_values)
            time.sleep(1)


            state, reward, done, info = env_spec.step(action)
            agent.test_reward(reward)

            if done:
                agent.end_episode(state)
                screen.clear()
                s = env_spec.render(mode='ansi')
                screen.addstr(s.getvalue())
                screen.addstr("End Episode\n")
                screen.refresh()
                time.sleep(1)
                break

    env_spec.close()
