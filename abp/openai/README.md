# Environments

This package contains code for custom environments and wrappers built on top on Open AI gym.

To use the custom environments add the following code:


```

import abp

env = gym.make("FruitCollection-v0")

```

The package currently has support for the following custom environments:

* Fruit Collection (FruitCollection-v0)
* Tic Tac Toe (TicTacToe-v0)
* Yahtzee (Yahtzee-v0)


# Wrappers

## Reward Wrapper

Adds support for decomposable reward types. The environment has to support reward decomposition
for this to work.

```

import abp

env = gym.make("FruitCollection-v0")
env = RewardWrapper(env)

state, reward, _, _ = env.step(action)
# The reward will contain decomposed reward types by default

state, reward, _, _ = env.step(action, decompose_reward = False)
# The reward will contain non decomposed reward 

```
