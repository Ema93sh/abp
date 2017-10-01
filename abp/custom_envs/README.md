# Environments

This package contains code for custom environments build on top on Open AI gym. To use the custom environments add the following code:

```
import abp.envs

env = gym.make("FruitCollection-v0")
```

The envs package currently has support for the following custom environments:

* Fruit Collection (FruitCollection-v0)
* Tic Tac Toe (TicTacToe-v0)
