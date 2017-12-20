from gym import Wrapper

import inspect
import logging
from functools import partial

logger = logging.getLogger(__name__)

class RewardWrapper(Wrapper):

    """

    Adds support for reward decomposition of the environment.
    To support reward decompisition the environment has to return reward as a list of dictionary values
    Example:
    [
        {
          "id": "unique ID of the type",
          "value": integer value indicating the reward got for this type,
          "description": description of the type (Can be used for explanation)
        }
    ]

    By default returns decomposed reward. It can be turned off by setting decompose_reward = False

    """

    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

    def step(self, action, decompose_reward = True):
        args , _ = inspect.getargspec(self.unwrapped._step, )
        if "decompose_reward" in args:
            self.unwrapped._step = partial(self.unwrapped._step, decompose_reward = decompose_reward)
        return self.env.step(action)
