from pysc2.lib import features


_PLAYER_SELF = features.PlayerRelative.SELF


class RewardWrapper(object):
    """docstring for RewardWrapper."""

    def __init__(self, obs, reward_types):
        super(RewardWrapper, self).__init__()
        self.reward_types = reward_types
        self.obs = obs

    def reward(self, next_obs):
        marines = [unit for unit in next_obs[-1].observation.feature_units
                   if unit.alliance == _PLAYER_SELF]
        selected_marines = [marine for marine in marines if marine.is_selected]
        selected_locs = [(marine.x, marine.y) for marine in selected_marines]

        reward = sum([obs.reward for obs in next_obs])

        decomposed_reward = {}

        for reward_type in self.reward_types:
            if reward_type in selected_locs:
                decomposed_reward[reward_type] = reward
            else:
                decomposed_reward[reward_type] = 0

        self.obs = next_obs

        return decomposed_reward
