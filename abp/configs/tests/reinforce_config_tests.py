import unittest
from abp.configs import ReinforceConfig
import os

file_path = os.path.dirname(os.path.abspath(__file__))
filename = "assets/reinforce.yml"
test_file = os.path.join(file_path, filename)


class ReinforceConfigTests(unittest.TestCase):
    """Unit tests for config objects"""

    def test_should_create_reinforce_config_object_from_file(self):
        reinforce_config = ReinforceConfig.load_from_yaml(test_file)

        self.assertEqual(reinforce_config.decay_steps, 500)
        self.assertEqual(reinforce_config.starting_epsilon, 0.9)
        self.assertEqual(reinforce_config.decay_rate, 0.80)
        self.assertEqual(reinforce_config.discount_factor, 1)
        self.assertEqual(reinforce_config.batch_size, 10)
        self.assertEqual(reinforce_config.memory_size, 20000)
        self.assertEqual(reinforce_config.summaries_path, "path/to/reinforcement/summaries.ckpt")

    def test_should_have_default_values(self):
        reinforce_config = ReinforceConfig()

        self.assertEqual(reinforce_config.decay_steps, 250)
        self.assertEqual(reinforce_config.starting_epsilon, 1.0)
        self.assertEqual(reinforce_config.decay_rate, 0.96)
        self.assertEqual(reinforce_config.discount_factor, 0.95)
        self.assertEqual(reinforce_config.batch_size, 35)
        self.assertEqual(reinforce_config.memory_size, 10000)
        self.assertEqual(reinforce_config.summaries_path, None)

    def test_should_be_able_to_set_property(self):
        reinforce_config = ReinforceConfig()

        self.assertEqual(reinforce_config.decay_steps, 250)

        reinforce_config.decay_steps = 100

        self.assertEqual(reinforce_config.decay_steps, 100)


if __name__ == '__main__':
    unittest.main()
