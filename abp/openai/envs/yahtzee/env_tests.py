import unittest
from mock import Mock

from env import YahtzeeEnv


class TestYahtzeeEnvScore(unittest.TestCase):

    def setUp(self):
        self.env = YahtzeeEnv()

    def test_roll_all_dice(self):
        # TODO: will probably fail, very less probability
        current_hand = self.env.current_hand
        next_hand = self.env.roll_dice()
        self.assertNotEqual(current_hand, next_hand)

    def test_not_roll_dice(self):
        # TODO: will probably pass for wrong for wrong reason
        current_hand = self.env.current_hand
        next_hand = self.env.roll_dice([1, 1, 1, 1, 1])
        self.assertEqual(current_hand, next_hand)

    def test_hold_some_dice(self):
        current_hand = self.env.current_hand
        next_hand = self.env.roll_dice([1, 1, 0, 0, 1])

        self.assertEqual(current_hand[0], next_hand[0])
        self.assertEqual(current_hand[1], next_hand[1])
        self.assertEqual(current_hand[4], next_hand[4])

    def test_score_upper_section(self):
        # ones
        self.env.current_hand = [1, 5, 1, 6, 1]
        score = self.env.score_upper_section(1)
        self.assertEqual(3, score)

        # Twoes
        self.env.current_hand = [2, 5, 1, 2, 1]
        score = self.env.score_upper_section(2)
        self.assertEqual(4, score)

        # Threes
        self.env.current_hand = [3, 3, 1, 2, 3]
        score = self.env.score_upper_section(3)
        self.assertEqual(9, score)

        # Fours
        self.env.current_hand = [4, 4, 1, 4, 4]
        score = self.env.score_upper_section(4)
        self.assertEqual(16, score)

        # Fives
        self.env.current_hand = [4, 4, 1, 4, 5]
        score = self.env.score_upper_section(5)
        self.assertEqual(5, score)

        # Sixes
        self.env.current_hand = [6, 6, 6, 6, 6]
        score = self.env.score_upper_section(6)
        self.assertEqual(30, score)

    def test_score_three_of_a_kind(self):
        self.env.current_hand = [2, 1, 2, 6, 2]
        score = self.env.of_a_kind(3)
        self.assertEqual(13, score)

        self.env.current_hand = [2, 1, 2, 6, 1]
        score = self.env.of_a_kind(3)
        self.assertEqual(0, score)

        self.env.current_hand = [1, 1, 1, 6, 1]
        score = self.env.of_a_kind(3)
        self.assertEqual(10, score)

    def test_score_four_of_a_kind(self):
        self.env.current_hand = [1, 1, 1, 6, 1]
        score = self.env.of_a_kind(4)
        self.assertEqual(10, score)

        self.env.current_hand = [1, 1, 1, 1, 1]
        score = self.env.of_a_kind(4)
        self.assertEqual(5, score)

        self.env.current_hand = [1, 1, 2, 6, 1]
        score = self.env.of_a_kind(4)
        self.assertEqual(0, score)

    def test_score_full_house(self):
        self.env.current_hand = [1, 1, 6, 6, 1]
        score = self.env.full_house()
        self.assertEqual(25, score)

        self.env.current_hand = [1, 2, 6, 6, 1]
        score = self.env.full_house()
        self.assertEqual(0, score)

        self.env.current_hand = [1, 6, 6, 6, 6]
        score = self.env.full_house()
        self.assertEqual(0, score)

    def test_score_small_straight(self):
        self.env.current_hand = [1, 6, 6, 6, 6]
        score = self.env.small_straight()
        self.assertEqual(0, score)

        self.env.current_hand = [4, 3, 3, 5, 2]
        score = self.env.small_straight()
        self.assertEqual(30, score)

        self.env.current_hand = [1, 6, 3, 4, 2]
        score = self.env.small_straight()
        self.assertEqual(30, score)

        self.env.current_hand = [1, 2, 3, 4, 5]
        score = self.env.small_straight()
        self.assertEqual(30, score)

    def test_score_large_straight(self):
        self.env.current_hand = [1, 2, 3, 4, 6]
        score = self.env.large_straight()
        self.assertEqual(0, score)

        self.env.current_hand = [4, 6, 3, 5, 2]
        score = self.env.large_straight()
        self.assertEqual(40, score)

        self.env.current_hand = [1, 5, 3, 4, 2]
        score = self.env.large_straight()
        self.assertEqual(40, score)

        self.env.current_hand = [1, 2, 3, 4, 5]
        score = self.env.large_straight()
        self.assertEqual(40, score)

    def test_score_chance(self):
        self.env.current_hand = [1, 6, 6, 6, 6]
        score = self.env.chance()
        self.assertEqual(25, score)

        self.env.current_hand = [1, 1, 1, 6, 1]
        score = self.env.chance()
        self.assertEqual(10, score)

    def test_score_yhatzee(self):
        self.env.current_hand = [1, 1, 1, 6, 1]
        score = self.env.yahtzee()
        self.assertEqual(0, score)

        self.env.current_hand = [1, 1, 1, 1, 1]
        score = self.env.yahtzee()
        self.assertEqual(50, score)

        self.env.current_hand = [6, 6, 6, 6, 6]
        score = self.env.yahtzee()
        self.assertEqual(50, score)

    def test_invalid_select_category(self):
        score = self.env.select_category(-1)
        self.assertEqual(-100, score)

        score = self.env.select_category(13)
        self.assertEqual(-100, score)

        self.env.categories[0] = 1
        score = self.env.select_category(0)
        self.assertEqual(-100, score)

        self.env.categories[12] = 1
        self.env.category_score[12] = 50
        score = self.env.select_category(12)
        self.assertNotEqual(-100, score)

    def test_add_upper_section_bonus(self):
        self.env.categories = [1] * 6 + [0] * 7
        self.env.category_score = {
            0: 5,
            1: 10,
            2: 15,
            3: 20,
            4: 25,
            5: 30
        }

        score = self.env.add_upper_section_bonus()
        self.assertEqual(score, 35)

        score = self.env.add_upper_section_bonus()
        self.assertEqual(score, 0)

    def test_valid_select_category(self):
        for i in range(6):
            expected_score = self.env.score_upper_section(i + 1)
            score = self.env.select_category(i)
            self.assertEqual(expected_score, score)

        for i in range(6, 13):
            expected_score = self.env.score_lower_section(i)
            score = self.env.select_category(i)
            self.assertEqual(expected_score, score, "%d : %d != %d" % (i, expected_score, score))

    def test_generate_state(self):
        self.env.current_hand = [1, 2, 3, 4, 5]
        self.env.categories = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.env.current_turn = 2

        actual_state = self.env.generate_state()
        expected_state = [0.1, 0.2, 0.3, 0.4, 0.5] + self.env.categories + [self.env.current_turn]
        self.assertEqual(expected_state, actual_state)

    def tearDown(self):
        self.env._reset()


class TestYahtzeeEnvGame(unittest.TestCase):
    """Test the Yahtzee Game"""

    def setUp(self):
        self.env = YahtzeeEnv()
        self.roll_dice_mock = Mock()
        self.roll_dice_mock.return_value = [1, 1, 1, 1, 1]
        self.env.roll_dice = self.roll_dice_mock

    def skip_step(self, times):
        for i in range(times):
            self.env._step(([0] * 5, 1))

    def test_single_step(self):
        state, reward, done, info = self.env._step(([0] * 5, 1))

        self.assertEqual(reward, 0)
        self.assertEqual(done, False)

    def test_three_steps(self):
        self.assertEqual(self.env.current_turn, 0)

        _, reward, done, _ = self.env._step(([0] * 5, 1))
        self.assertEqual(reward, 0)
        self.assertEqual(self.env.current_turn, 1)
        self.assertEqual(self.env.categories, [0] * 13)

        self.assertEqual(done, False)

        _, reward, done, _ = self.env._step(([0] * 5, 1))
        self.assertEqual(reward, 0)
        self.assertEqual(self.env.current_turn, 2)
        self.assertEqual(self.env.categories, [0] * 13)
        self.assertEqual(self.env.current_hand, [1, 1, 1, 1, 1])
        self.assertEqual(done, False)

        _, reward, done, _ = self.env._step(([0] * 5, 1))
        self.assertEqual(reward, 0)
        self.assertEqual(self.env.current_turn, 3)
        self.assertEqual(self.env.categories, [0] * 13)
        self.assertEqual(self.env.current_hand, [1, 1, 1, 1, 1])
        self.assertEqual(done, False)

        state, reward, done, info = self.env._step(([0] * 5, 0))
        expected_reward = 5
        self.assertEqual(self.env.current_turn, 0)
        self.assertEqual(expected_reward, reward)
        self.assertEqual(self.env.current_hand, [1, 1, 1, 1, 1])
        self.assertEqual(done, False)

        state, reward, done, info = self.env._step(([0] * 5, 0))
        self.assertEqual(self.env.current_turn, 1)
        self.assertEqual(0, reward)
        self.assertEqual(done, False)

    def test_end_game_when_invalid_category(self):
        self.skip_step(3)
        self.assertEqual(self.env.current_turn, 3)

        _ = self.env._step(([0] * 5, 0))

        self.assertEqual(self.env.categories, [1] + [0] * 12)

        self.skip_step(3)

        self.assertEqual(self.env.current_turn, 3)

        self.assertEqual(self.env.categories, [1] + [0] * 12)

        _, reward, done, _ = self.env._step(([0] * 5, 0))

        self.assertEqual(-100, reward)
        self.assertEqual(done, True)

    def test_single_episode(self):
        reward = 0

        for i in range(12):
            self.skip_step(3)
            _, r, done, _ = self.env._step(([0] * 5, i))
            reward += r
            self.assertEqual(done, False)

        self.skip_step(3)

        _, r, done, _ = self.env._step(([0] * 5, 12))
        reward += r

        self.assertEqual(done, True)
        self.assertEqual(reward, 70)

    def test_state_for_a_step(self):
        state, _, _, _ = self.env._step(([0] * 5, 0))
        self.assertEqual([0.1, 0.1, 0.1, 0.1, 0.1] + [0] * 13 + [1], state)

    def test_episode_with_upper_section_bonus_score(self):
        for i in range(6):
            self.roll_dice_mock.return_value = [i + 1] * 5
            self.skip_step(3)
            _, r, done, _ = self.env._step(([0] * 5, i))
            if i != 5:
                self.assertEqual(r, (i + 1) * 5)
            else:
                self.assertEqual(r, 6 * 5 + 35)

    def test_episode_without_upper_section_bonus_score(self):
        for i in range(6):
            self.roll_dice_mock.return_value = [i + 1] * 5
            self.skip_step(3)
            _, r, done, _ = self.env._step(([0] * 5, i))
            if i != 5:
                self.assertEqual(r, (i + 1) * 5)
            else:
                self.assertEqual(r, 6 * 5 + 35)

    def test_episode_with_yahtzee_bonus_score(self):
        self.roll_dice_mock.return_value = [1] * 5
        self.skip_step(3)
        _, r, _, _ = self.env._step(([0] * 5, 12))
        self.assertEqual(r, 50)

        self.skip_step(3)
        _, r, done, _ = self.env._step(([0] * 5, 12))
        self.assertEqual(r, 100)
        self.assertEqual(done, False)

    def test_episode_without_yahtzee_bonus_score(self):
        self.roll_dice_mock.return_value = [1, 2, 3, 4, 5]
        self.skip_step(3)
        _, r, _, _ = self.env._step(([0] * 5, 12))
        self.assertEqual(r, 0)

        self.roll_dice_mock.return_value = [1] * 5

        self.skip_step(3)
        _, r, done, _ = self.env._step(([0] * 5, 12))
        self.assertEqual(r, -100)
        self.assertEqual(done, True)

        self.skip_step(3)
        _, r, done, _ = self.env._step(([0] * 5, 12))
        self.assertEqual(r, -100)
        self.assertEqual(done, True)

    def tearDown(self):
        self.env._reset()


if __name__ == '__main__':
    unittest.main()
