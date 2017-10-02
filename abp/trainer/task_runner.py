import argparse
from importlib import import_module
import abp
import logging


task_map = {
    "cartpole": {
        "dqn": "abp.examples.open_ai.cart_pole.cart_pole_dqn",
        "hra": "abp.examples.open_ai.cart_pole.cart_pole_hra"
    },
    "fruitcollection": {
        "dqn": "abp.examples.open_ai.custom_envs.fruit_collection_dqn",
        "hra": "abp.examples.open_ai.custom_envs.fruit_collection_hra",
        "user": "abp.examples.open_ai.custom_envs.fruit_collection_user"
    },
    "tictactoe": {
        "dqn": "abp.examples.open_ai.custom_envs.tic_tac_toe_dqn",
        "hra": "abp.examples.open_ai.custom_envs.tic_tac_toe_hra",
        "user": "abp.examples.open_ai.custom_envs.tic_tac_toe_user"
    }
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


    args = parser.parse_args()


    logging.info("Running: " + args.example + " with adaptive: " + args.adaptive)

    task = task_map[args.example][args.adaptive]

    task_module = import_module(task)



    task_module.run_task(
        args.job_dir,
        render = args.render,
        model_path = args.model_path,
        restore_model = args.restore_model,
        training_episode = args.training_episodes,
        test_episodes = args.test_episodes,
        decay_steps = args.decay_steps
    )


if __name__ == '__main__':
    main()
