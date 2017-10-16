import gym
import numpy as np
import abp.custom_envs

import time
import curses

screen = curses.initscr()
curses.savetty()
curses.noecho()
curses.cbreak()
curses.curs_set(0)
# screen.nodelay(0)
screen.keypad(1)

def run_task(config):
    env_spec = gym.make("TicTacToe-v0")
    max_episode_steps = env_spec._max_episode_steps

    state = env_spec.reset()

    for epoch in range(config.test_episodes):
        state = env_spec.reset()
        reward = 0
        for steps in range(max_episode_steps):
            screen.clear()
            screen.addstr("reward:" + str(reward) + "\n")
            screen.addstr("State:\n" + str(state) + "\n\n")
            s = env_spec.render(mode='ansi')
            screen.addstr(s.getvalue())
            screen.refresh()
            key = screen.getch()
            action = int(key) - 48

            state, reward, done, info = env_spec.step(action)

            if done:
                screen.clear()
                s = env_spec.render(mode='ansi')
                screen.addstr(s.getvalue())
                if info['illegal_move']:
                    screen.addstr("Lost Cause of illegal move\n")
                elif info['x_won'] == True:
                    screen.addstr("You Won\n")
                elif info['o_won'] == True:
                    screen.addstr("Opponent Won\n")
                else:
                    screen.addstr("Draw\n")
                screen.refresh()
                time.sleep(2)
                break

    env_spec.close()
