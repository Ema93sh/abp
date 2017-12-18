import gym
import numpy as np
import collections

from gym import spaces
from functools import partial

class YahtzeeEnv(gym.Env):
    """ Open AI Env for Yahtzee Game """

    metadata = {'render.modes': ['ansi']}

    #TODO: bonus score if the upper section is above the threshold
    # Categories
    # 0 - Ones
    # 1 - Twoe
    # 2 - Thress
    # 3 - Fours
    # 4 - Fives
    # 5 - Sixes
    # 6 - Three of a kind
    # 7 - Four of a kind
    # 8 - Full House
    # 9 - Small Straight
    # 10 - Large Straight
    # 11 - Chance
    # 12 - Yahtzee


    def __init__(self):
        super(YahtzeeEnv, self).__init__()
        self.current_hand = self.roll_dice()
        self.categories = [0] * 13
        self.current_turn = 0
        self.current_category_turn = 0
        self.upper_section_bonus = False
        self.yahtzee_bonus = True

        self.category_score = {}

        self.score_registry = {
         6: partial(self.of_a_kind, 3),
         7: partial(self.of_a_kind, 4),
         8: self.full_house,
         9: self.small_straight,
         10: self.large_straight,
         11: self.chance,
         12: self.yahtzee
        }

    def roll_dice(self, holds = [0, 0, 0, 0, 0]):
        dice = []
        for i in range(5):
            if holds[i] == 0:
                dice.append(np.random.choice(range(1, 7)))
            else:
                dice.append(self.current_hand[i])
        return dice

    def generate_state(self):
        return self.current_hand + self.categories + [self.current_turn]

    def info(self):
        return {"current_hand": self.current_hand, "categories": self.categories, "turn": self.current_turn}

    def _reset(self):
        self.current_hand = []
        self.categories = [0] * 13
        self.current_turn = 0
        self.current_category_turn = 0
        self.upper_section_bonus = False
        pass

    def score_upper_section(self, kind):
        return sum(filter(lambda x: x == kind, self.current_hand))

    def score_lower_section(self, category):
        return self.score_registry[category]()

    def of_a_kind(self, count):
        counts = collections.Counter(self.current_hand)
        return sum(self.current_hand) if any([ i >= count for i in counts.values()]) else 0

    def full_house(self):
        counts = collections.Counter(self.current_hand)
        common = counts.most_common(2)
        if len(common) < 2:
            return 0
        [l, r] = common
        if l[1] == 3 and r[1] == 2:
            return 25
        return 0

    def small_straight(self):
        return 30 if self.longest_sequence() >= 4 else 0

    def large_straight(self):
        return 40 if self.longest_sequence() >= 5 else 0

    def yahtzee(self):
        counts = collections.Counter(self.current_hand)
        [(_, c)] = counts.most_common(1)
        if c == 5: #TODO: Arrggghh! too many ifssss
            if self.categories[12] >= 1:
                return 0 if self.category_score[12] == 0 else 100
            return 50
        else:
            return 0
        return 50 if c == 5 else 0

    def chance(self):
        return sum(self.current_hand)

    def add_upper_section_bonus(self):
        if self.upper_section_bonus:
            return 0

        if sum(self.categories[:6]) == 6:
            s = 0
            for i in range(6):
                s += self.category_score[i]
            if s > 63:
                self.upper_section_bonus = True
                return 35

        return 0

    def longest_sequence(self):
        hand = set(self.current_hand)
        max_sequence = 0
        for i in hand:
            if i - 1 not in hand:
                num = i
                current_length = 1
                while num + 1 in hand:
                    num += 1
                    current_length += 1
                max_sequence = max(max_sequence, current_length)
        return max_sequence

    def select_category(self, category):
        if category < 0 or category >= 13:
            return -1000 # Invalid category

        if category != 12 and self.categories[category]:
            return -1000 #Category already choosen and its not yahtzee

        if category in range(6):
            return self.score_upper_section(category + 1)
        else:
            return self.score_lower_section(category)


    def _step(self, holds, category):
        reward = 0
        info = {}
        done = False

        if self.current_turn  == 3:
            reward = self.select_category(category)
            self.category_score[category] = reward if category not in self.category_score else reward + self.category_score[category]
            self.categories[category] += 1
            self.current_turn = 0
            self.current_category_turn += 1
        else:
            self.current_hand = self.roll_dice(holds)
            self.current_turn += 1

        reward += self.add_upper_section_bonus()

        if self.current_category_turn == 13:
            done = True

        if reward == -1000: # End when you choose an invalid category
            done = True

        return self.generate_state(), reward, done, self.info()


    def render_ansi(self):
        outfile = StringIO()
        outfile.write("Category | Score")

        for category in range(13):
            score = self.category_score[category] if category in self.category_score else 0
            outfile.write("%-9d | %-5d \n" % (i + 1, score))

        outfile.write("Current Dice: " + str(self.current_hand) + "\n")

        return outfile

    def _render(self, mode = 'human', close = False):
        if close:
            return None
        return self.render_ansi()
