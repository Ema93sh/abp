import gym
import numpy as np
import collections

from six import StringIO
from functools import partial


class YahtzeeEnv(gym.Env):
    """ Open AI Env for Yahtzee Game """

    metadata = {'render.modes': ['human', 'ansi']}

    # TODO: bonus score if the upper section is above the threshold
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
    # 13 - No category (used when its not current turn)

    def __init__(self):
        super(YahtzeeEnv, self).__init__()
        self.score_registry = {
            6: partial(self.of_a_kind, 3),
            7: partial(self.of_a_kind, 4),
            8: self.full_house,
            9: self.small_straight,
            10: self.large_straight,
            11: self.chance,
            12: self.yahtzee
        }

        self.category_map = {
            0: "Ones",
            1: "Twoes",
            2: "Threes",
            3: "Fours",
            4: "Fives",
            5: "Sixes",
            6: "3 of a kind",
            7: "4 of a kind",
            8: "Full House",
            9: "Small Straight",
            10: "Large Straight",
            11: "Chance",
            12: "Yahtzee",
            13: "Bonus"
        }

        self._reset()

    def roll_dice(self, holds=[0, 0, 0, 0, 0]):
        dice = []
        for i in range(5):
            if holds[i] == 0:
                dice.append(np.random.choice(range(1, 7)))
            else:
                dice.append(self.current_hand[i])
        return dice

    def generate_state(self):
        return map(lambda x: x / 10.0, self.current_hand) + self.categories + [self.current_turn]

    def info(self):
        return {"current_hand": self.current_hand,
                "categories": self.categories,
                "turn": self.current_turn,
                "category_score": self.category_score}

    def _reset(self):
        self.current_hand = self.roll_dice()
        self.categories = [0] * 13
        self.current_turn = 0
        self.current_category_turn = 0
        self.upper_section_bonus = False
        self.category_score = {}
        return self.generate_state()

    def score_upper_section(self, kind):
        return sum(filter(lambda x: x == kind, self.current_hand))

    def score_lower_section(self, category):
        return self.score_registry[category]()

    def of_a_kind(self, count):
        counts = collections.Counter(self.current_hand)
        return sum(self.current_hand) if any([i >= count for i in counts.values()]) else 0

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
        if c == 5:  # TODO: Arrggghh! too many ifssss
            if self.categories[12] >= 1:
                return -100 if self.category_score[12] <= 0 else 100
            return 50
        else:
            return 0
        return 50 if c == 5 else 0

    def chance(self):
        return sum(self.current_hand)

    def add_upper_section_bonus(self):
        if self.upper_section_bonus:
            return 0

        if all(map(lambda x: x == 1, self.categories[:6])):
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
            return -100  # Invalid category

        if category != 12 and self.categories[category] > 0:
            return -100  # Category already choosen and its not yahtzee

        if category in range(6):
            return self.score_upper_section(category + 1)
        else:
            return self.score_lower_section(category)

    def _step(self, action, decompose_reward=False):
        holds, category = action
        reward = 0
        # d_reward = [0] * self.
        done = False

        if self.current_turn == 3:
            reward = self.select_category(category)
            score = reward if category not in self.category_score else reward + \
                self.category_score[category]
            self.category_score[category] = score
            self.categories[category] += 1
            self.current_turn = 0
            self.current_category_turn += 1
            self.current_hand = self.roll_dice()

            if self.current_category_turn >= 13:
                done = True
        else:
            self.current_hand = self.roll_dice(holds)
            self.current_turn += 1

        bonus = self.add_upper_section_bonus()
        if bonus > 0:
            self.category_score[13] = bonus

        reward += bonus

        if reward < 0:  # End when you choose an invalid category
            done = True

        return self.generate_state(), reward, done, self.info()

    def render_ansi(self):
        outfile = StringIO()
        outfile.write("%-25s | %-5s\n" % ("Category", "Score"))
        outfile.write("-" * 35 + "\n")

        for category in range(14):
            score = str(self.category_score[category]
                        ) if category in self.category_score else " -- "
            outfile.write("%-25s | %-5s \n" % (self.category_map[category], score))

        outfile.write("-" * 35 + "\n")
        outfile.write("Total Score: " + str(sum(self.category_score.values())) + "\n")
        outfile.write("-" * 35 + "\n\n\n")
        outfile.write("Current Hand: " + str(self.current_hand) + "\n")
        outfile.write("Current Turn: " + str(self.current_turn) + "\n")

        return outfile

    def _render(self, mode='human', close=False):
        if close:
            return None
        return self.render_ansi()
