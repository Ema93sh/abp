# A hunter is searching for a treasure! :| But WHY???????????

import random
import os
from collections import Counter

import numpy as np
import gym
import visdom
import time

from .env_map import EnvMap


class FruitCollectionEnv(gym.Env):
    """The agent is looking for fruit without getting hit by lightning"""

    def __init__(self, map_name="10x10_easy", state_representation="linear"):
        self.action_space = 4
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.vis = visdom.Visdom()
        self.name = "FruitCollection"
        self.map = EnvMap(os.path.join(self.current_dir, "maps", map_name + ".map"))
        self.max_step = 100
        self.treasure_locations = self.map.get_all_treasure_locations()
        self.total_treasure = len(self.treasure_locations)
        self.image_window = None
        self.heatmap_window = None
        self.action_names = ['Up', 'Right', 'Down', 'Left']
        self.reward_types = ["(%d, %d)" % (location)
                             for location in self.treasure_locations] + ["Lightning Strike"]
        self.state_representation = state_representation
        self.reset()

    def reset(self, agent_location=None, treasure_found=None, state_representation="linear"):
        self.state_representation = state_representation
        self.agent_location = agent_location if agent_location else self.map.agent_location()
        self.current_step = 0
        self.treasure_found = treasure_found if treasure_found else [
            False for i in self.treasure_locations]

        if len(self.treasure_found) != len(self.treasure_locations):
            print(len(self.treasure_found), len(self.treasure_locations))
            raise Exception("Treasure found dim and treasure location dim should be the same!")

        self.lightning_pos = []
        self.total_reward = 0
        self.current_reward = []

        return self.generate_state()

    def move(self, action):
        agent_pos = None
        if action == 0:
            agent_pos = (self.agent_location[0] - 1, self.agent_location[1])
        elif action == 1:
            agent_pos = (self.agent_location[0], self.agent_location[1] + 1)
        elif action == 2:
            agent_pos = (self.agent_location[0] + 1, self.agent_location[1])
        elif action == 3:
            agent_pos = (self.agent_location[0], self.agent_location[1] - 1)

        if self.map.has_wall(agent_pos[0], agent_pos[1]):
            return

        self.agent_location = agent_pos

    def generate_state(self):
        agent_grid = np.zeros(self.map.shape())
        agent_grid[self.agent_location[0], self.agent_location[1]] = 1

        treasure_locations = np.zeros(self.total_treasure)
        treasure_locations[[not x for x in self.treasure_found]] = 1

        lightning_grid = np.zeros(self.map.shape())
        for pos in self.lightning_pos:
            lightning_grid[pos] = 1

        if self.state_representation == "rgb":
            treasure_locations = np.zeros(self.map.shape())
            for pos in self.treasure_locations:
                treasure_locations[pos] = 1
            state = np.stack((agent_grid, treasure_locations, lightning_grid))
        else:
            state = self.__get_obs_image()

        return state

    def generate_lightning(self):
        row, col = self.map.shape()

        lightning_pos = []
        lightning_probability = self.map.get_all_lightning_probability()

        for i in range(row):
            for j in range(col):
                probability = lightning_probability[i][j]
                if random.random() < probability:
                    lightning_pos.append((i, j))

        return lightning_pos

    def render(self):
        if self.vis is None:
            return

        obs_image = self.__get_obs_image()

        total_treasure_found = Counter(self.treasure_found)[True]

        title = '{}\t\tTreasures Found:{} Overall_Reward:{} Step Reward:{} Steps:{}' \
            .format(self.name, total_treasure_found, self.total_reward,
                    self.current_reward,
                    self.current_step)

        opts = dict(title=title, width=360, height=350)
        lightning_probability = np.array(self.map.get_all_lightning_probability()[::-1])

        heatmap_opts = dict(xtick=False, ytick=False, ytickmin=0, ytickmax=0)

        if self.heatmap_window is None:
            self.heatmap_window = self.vis.heatmap(lightning_probability, opts=heatmap_opts)
        else:
            self.vis.heatmap(lightning_probability, win=self.heatmap_window, opts=heatmap_opts)

        if self.image_window is None:
            self.image_window = self.vis.image(obs_image, opts=opts)
        else:
            self.vis.image(obs_image, opts=opts, win=self.image_window)

    def __get_obs_image(self):
        shape = self.map.shape()
        img = np.zeros((3,) + shape)
        img[:] = 255

        # Treasure
        for i, consumed in enumerate(self.treasure_found):
            row, col = self.treasure_locations[i]

            if not consumed:
                img[0, row, col] = 91
                img[1, row, col] = 226
                img[2, row, col] = 116
            else:
                img[0, row, col] = 0
                img[1, row, col] = 0
                img[2, row, col] = 0

        # lightning
        for row, col in self.lightning_pos:
            img[0, row, col] = 238
            img[1, row, col] = 232
            img[2, row, col] = 170

        # wall
        for row, col in self.map.get_all_wall_locations():
            img[0, row, col] = 205
            img[1, row, col] = 133
            img[2, row, col] = 63

        # color agent
        row, col = self.agent_location
        img[0, row, col] = 224
        img[1, row, col] = 80
        img[2, row, col] = 20
        return img

    def step(self, action, decompose_reward=False, log_time=False):
        self.current_step += 1

        rewards = {}

        for reward_type in self.reward_types:
            rewards[reward_type] = 0

        self.move(action)

        start_time = time.time()
        self.lightning_pos = self.generate_lightning()
        end_time = time.time()

        if self.agent_location in self.treasure_locations:
            index = self.treasure_locations.index(self.agent_location)
            if index != -1 and not self.treasure_found[index]:
                reward_type = self.reward_types[index]
                rewards[reward_type] = 2
                self.treasure_found[index] = True

        struck_by_lightning = self.agent_location in self.lightning_pos

        if struck_by_lightning:
            rewards["Lightning Strike"] = -1

        all_treasure_collected = all(self.treasure_found)
        done = ((self.current_step >= self.max_step) or
                all_treasure_collected or
                struck_by_lightning)

        self.total_reward += sum(rewards.values())
        self.current_reward = rewards

        rewards = rewards if decompose_reward else sum(rewards.values())

        state = self.generate_state()

        if log_time:
            print("Generate state time %.2f" % (end_time - start_time))
        info = {"state_gen_time": end_time - start_time, "lightning_pos": self.lightning_pos}

        return state, rewards, done, info

    def close(self):
        pass


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
