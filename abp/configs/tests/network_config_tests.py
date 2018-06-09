import unittest
from abp.configs import NetworkConfig
import os

file_path = os.path.dirname(os.path.abspath(__file__))
filename = "assets/network.yml"
test_file = os.path.join(file_path, filename)


class NetworkConfigTests(unittest.TestCase):
    """Unit tests for config objects"""

    def test_should_create_network_config_object_from_file(self):
        network_config = NetworkConfig.load_from_yaml(test_file)

        self.assertEqual(network_config.input_shape, [20])
        self.assertEqual(network_config.layers, [50, 50])
        self.assertEqual(network_config.output_shape, [5])

        self.assertEqual(network_config.restore_network, False)
        self.assertEqual(network_config.network_path, "test/the/network/path.ckpt")

        self.assertEqual(network_config.summaries_path, "test/summaries/path.ckpt")
        self.assertEqual(network_config.summaries_step, 50)

        self.assertEqual(network_config.shared_layers, [250])
        self.assertEqual(network_config.aggregator, "average")

    def test_should_have_default_values(self):
        network_config = NetworkConfig()

        self.assertEqual(network_config.input_shape, [10])
        self.assertEqual(network_config.layers, [100, 100])
        self.assertEqual(network_config.output_shape, [5])

        self.assertEqual(network_config.restore_network, True)
        self.assertEqual(network_config.network_path, None)

        self.assertEqual(network_config.summaries_path, None)
        self.assertEqual(network_config.summaries_step, 100)

        self.assertEqual(network_config.shared_layers, [])
        self.assertEqual(network_config.aggregator, "average")

    def test_should_be_able_to_set_property(self):
        network_config = NetworkConfig()

        self.assertEqual(network_config.input_shape, [10])

        network_config.input_shape = [5, 5]

        self.assertEqual(network_config.input_shape, [5, 5])

        self.assertEqual(network_config.layers, [100, 100])

        network_config.layers = [10]

        self.assertEqual(network_config.layers, [10])

    def test_should_be_able_to_access_nested_properties(self):
        network_config = NetworkConfig.load_from_yaml(test_file)

        self.assertEqual(len(network_config.networks), 4)

        up_network = network_config.networks[0]
        down_network = network_config.networks[1]

        self.assertEqual(up_network['input_shape'], [110])
        self.assertEqual(up_network['layers'], [100])
        self.assertEqual(up_network['output_shape'], [4])

        self.assertEqual(down_network['input_shape'], [110])
        self.assertEqual(down_network['layers'], [50, 50])
        self.assertEqual(down_network['output_shape'], [4])


if __name__ == '__main__':
    unittest.main()
