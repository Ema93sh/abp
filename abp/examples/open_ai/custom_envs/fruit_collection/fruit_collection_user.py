import gym
import numpy as np
import abp.custom_envs
import time

def run_task(config):
    env_spec = gym.make("FruitCollection-v0")
    max_episode_steps = env_spec._max_episode_steps

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
        state = env_spec.reset()
        for steps in range(max_episode_steps):
            screen.clear()
            s = env_spec.render(mode='ansi')
            screen.addstr(s.getvalue())
            screen.refresh()
            key = screen.getch()
            action = 0

            if key ==  259:
                screen.addstr("Up")
                action = 2
            elif key == 258:
                screen.addstr("Down")
                action = 3
            elif key == 260:
                screen.addstr("Left")
                action = 0
            elif key == 261:
                screen.addstr("Right")
                action = 1

            state, reward, done, info = env_spec.step(action)
            possible_fruit_locations = info["possible_fruit_locations"]
            collected_fruit = info["collected_fruit"]
            current_fruit_locations = info["current_fruit_locations"]

            r = None
            if collected_fruit is not None:
                r = possible_fruit_locations.index(collected_fruit)
                screen.addstr(str(possible_fruit_locations) + "\n")
                screen.addstr("Collected Fruit at location %d(%d)\n" % (collected_fruit, r))
                screen.refresh()
                time.sleep(5)


            if done or steps == (max_episode_steps - 1):
                break

    env_spec.close()
