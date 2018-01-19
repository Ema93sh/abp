import argparse
import os

from importlib import import_module

from abp.configs import NetworkConfig, ReinforceConfig, EvaluationConfig

#TODO: Need a better way to run a task.

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--folder',
        help = 'The folder containing the config files',
        required = True
    )

    # TODO Better way to load the task to run
    parser.add_argument(
        '-t', '--task',
        help = "The task to run. The python module cointaining the ABP program",
        required = True
    )

    parser.add_argument(
        '-j', '--job-dir',
        help = "Job dir",
        required = False
    )

    args = parser.parse_args()

    evaluation_config = EvaluationConfig.load_from_yaml(os.path.join(args.folder, "evaluation.yml"))

    network_config = NetworkConfig.load_from_yaml(os.path.join(args.folder, "network.yml"))

    reinforce_config = ReinforceConfig.load_from_yaml(os.path.join(args.folder, "reinforce.yml"))

    task_module = import_module(args.task)

    task_module.run_task(evaluation_config, network_config, reinforce_config)

    return 0

if __name__ == '__main__':
    main()
