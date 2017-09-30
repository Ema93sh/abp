import argparse
from importlib import import_module

task_map = {
    "cartpole": {
        "dqn": "abp.examples.open_ai.cart_pole.cart_pole_dqn",
        "hra": "abp.examples.open_ai.cart_pole.cart_pole_hra"
    },
    "fruitcollection": {
        "dqn": "abp.examples.open_ai.custom_envs.fruit_collection_dqn",
        "hra": "abp.examples.open_ai.custom_envs.fruit_collection_hra"
    },
    "tictactoe": {
        "dqn": "abp.examples.open_ai.custom_envs.tic_tac_toe_dqn",
        "hra": "abp.examples.open_ai.custom_envs.tic_tac_toe_hra"
    }
}

def main():
     parser = argparse.ArgumentParser()

     parser.add_argument(
      '-e', '--example',
      help = 'The example to run',
      nargs = 1,
      required = True)

     parser.add_argument(
     '-a', '--adaptive',
     help = 'The adaptive to use for the run',
     nargs = 1,
     required = True
     )

     args = parser.parse_args()

     example = args.example[0]
     adaptive = args.adaptive[0]


     task = task_map[example][adaptive]

     task_module = import_module(task)

     task_module.run_task()


if __name__ == '__main__':
    main()
