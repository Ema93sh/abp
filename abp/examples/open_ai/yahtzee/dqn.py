from abp import DQNAdaptive


def run_task(config):
    config.name = "Yahtzee-v0"
    env = YahtzeeState()

    max_episode_steps = 1000

    config.size_features = 18 # 5 (dice) + 13 (configs)
    config.action_size = 8 + 1 # 0 : roll, 1-8 : category

    agent = DQNAdaptive(config)
    print config.training_episode
    #Training Episodes
    for episode in config.training_episode:
        env.reinitialize()
        for step in max_episode_steps:
            if env.is_terminal():
                break
            state = generate_state(env) #get current state
            action = agent.predict(state)
            # action = generate_action(action)
            reward = env.take_action(action) #gets a reward vector
            import pdb; pdb.set_trace()



    agent.disable_learning()

    #Test Episodes
    # for episode in config.test_episodes:
    #     env.reinitialize()
    #     for step in max_episode_steps:
    #         if env.is_terminal():
    #             break
    #
    #         action = ???
    #         action_type = ??? #Choose whether to roll dice or not
    #         action_category = ??? #Choose category
    #
    #         reward = env.take_action(action) #gets a reward vector
