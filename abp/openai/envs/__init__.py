from gym.envs.registration import register


register(
    id='TicTacToe-v0',
    entry_point='abp.openai.envs.tic_tac_toe:TicTacToeEnv',
    max_episode_steps=1000,  # TODO
    reward_threshold=40  # TODO
)

register(
    id='FruitCollection-v0',
    entry_point='abp.openai.envs.fruit_collection:FruitCollectionEnv'
)

register(
    id='Traveller-v0',
    entry_point='abp.openai.envs.traveller:TravellerEnv',
    reward_threshold=12  # TODO
)

register(
    id='Yahtzee-v0',
    entry_point='abp.openai.envs.yahtzee:YahtzeeEnv',
    max_episode_steps=1000,  # TODO
    # https://blogs.msdn.microsoft.com/matthew_van_eerde/2011/11/30/what-is-a-perfect-score-in-yahtzee/
    reward_threshold=1505
)

register(
    id='WolfHunt-v0',
    entry_point='abp.openai.envs.wolf_hunt:WolfHuntEnv'
)
