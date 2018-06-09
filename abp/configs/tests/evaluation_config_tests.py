import unittest
from abp.configs import EvaluationConfig
import os

file_path = os.path.dirname(os.path.abspath(__file__))
filename = "assets/evaluation.yml"
test_file = os.path.join(file_path, filename)


class EvaluationConfigTests(unittest.TestCase):
    """Unit tests for config objects"""

    def test_should_create_evaluation_config_object_from_file(self):
        evaluation_config = EvaluationConfig.load_from_yaml(test_file)

        self.assertEqual(evaluation_config.name, "test")
        self.assertEqual(evaluation_config.env, "Test-v0")
        self.assertEqual(evaluation_config.summaries_path, "test/the/network/path.ckpt")
        self.assertEqual(evaluation_config.training_episodes, 1000)
        self.assertEqual(evaluation_config.test_episodes, 50)
        self.assertEqual(evaluation_config.render, True)

    def test_should_have_default_values(self):
        evaluation_config = EvaluationConfig()

        self.assertEqual(evaluation_config.name, "default name")
        self.assertEqual(evaluation_config.env, None)
        self.assertEqual(evaluation_config.summaries_path, None)
        self.assertEqual(evaluation_config.training_episodes, 100)
        self.assertEqual(evaluation_config.test_episodes, 100)
        self.assertEqual(evaluation_config.render, False)

    def test_should_be_able_to_set_property(self):
        evaluation_config = EvaluationConfig()

        self.assertEqual(evaluation_config.name, "default name")

        evaluation_config.name = "Set Test"

        self.assertEqual(evaluation_config.name,  "Set Test")


if __name__ == '__main__':
    unittest.main()
