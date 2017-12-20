import argparse
import logging

from importlib import import_module

import abp
from abp.utils import AdaptiveConfig

task_map = {
    "cartpole": "abp.examples.open_ai.gym.cart_pole",
    "fruitcollection": "abp.examples.open_ai.fruit_collection",
    "tictactoe": "abp.examples.open_ai.tic_tac_toe",
    "traveller": "abp.examples.open_ai.traveller",
    "yahtzee": "abp.examples.open_ai.yahtzee"
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-e', '--example',
        help = 'The example to run',
        required = True
    )

    parser.add_argument(
        '-a', '--adaptive',
        help = 'The adaptive to use for the run',
        required = True
    )

    parser.add_argument(
        '--job-dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )

    parser.add_argument(
        '-mp', '--model-path',
        help = "The location to save the model"
    )

    parser.add_argument(
        '--restore-model',
        help = 'Restore the model instead of training a new model',
        action = 'store_true'
    )

    parser.add_argument(
        '-r', '--render',
        help = 'Set if it should render the test episodes',
        action = 'store_true',
        default = False
    )

    parser.add_argument(
        '--training-episodes',
        help = "Set the number of training episodes",
        type = int,
        default = 500
    )

    parser.add_argument(
        '--test-episodes',
        help = "Set the number of test episodes",
        type = int,
        default = 100
    )

    parser.add_argument(
        '--decay-steps',
        help = "Set the decay rate for exploration",
        type = int,
        default = 250
    )

    parser.add_argument(
        '--gamma',
        help = "Set the discount factor",
        type = float,
        default = 0.99
    )

    parser.add_argument(
        '--memory-size',
        help = "Set the memory size for DQN",
        type = int,
        default = 10000
    )

    parser.add_argument(
        '--disable-learning',
        help = "Disable learning for the adaptive",
        default = False,
        action = 'store_true'
    )


    args = parser.parse_args()

    logging.info("Running: " + args.example + " with adaptive: " + args.adaptive)

    task = task_map[args.example] + "." + args.adaptive

    task_module = import_module(task)

    default_adaptive_config  = AdaptiveConfig(args)

    task_module.run_task(default_adaptive_config)


if __name__ == '__main__':
    main()
