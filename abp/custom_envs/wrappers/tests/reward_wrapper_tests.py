import unittest
import gym
import abp.custom_envs

from mock import Mock, patch
from abp.custom_envs.wrappers import RewardWrapper

class DecomposedRewardWrapperTests(unittest.TestCase):

    def setUp(self):
        self.env = gym.make("Yahtzee-v0")

    def test_should_call_env_with_reward_decomposition(self):
        with patch.object(self.env.unwrapped, '_step', wraps=self.env.unwrapped._step) as mock_method:
            wrapped_env = RewardWrapper(self.env)
            wrapped_env.reset()
            dummy_action = ([0] * 5, 0)
            wrapped_env.step(dummy_action, True)
            mock_method.assert_called_with(dummy_action, decompose_reward = True)

    def tearDown(self):
        del self.env
