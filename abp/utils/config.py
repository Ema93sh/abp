
class AdaptiveConfig(object):
    """Configuration for the adaptives"""
    def __init__(self, args):
        super(AdaptiveConfig, self).__init__()
        self.name =  'Default'
        self.action_size =  None #Required
        self.size_features =  None #Required
        self.size_rewards = None #Required
        self.job_dir =  args.job_dir
        self.model_path =  args.model_path
        self.restore_model =  args.restore_model
        self.decay_steps =  args.decay_steps
        self.replace_target_steps =  300 #TODO unused
        self.gamma =  args.gamma
        self.memory_size =  args.memory_size
        self.render =  args.render
        self.training_episode =  args.training_episodes
        self.test_episodes =  args.test_episodes
        self.starting_epsilon =  1.0
        self.epsilon_decay_rate =  0.96
        self.learning =  not args.disable_learning


class ModelConfig(object): #TODO
    """docstring for ModelConfig."""
    def __init__(self, args):
        super(ModelConfig, self).__init__()
        #Model Configuration TODO
        self.number_of_hidden_layers = args.number_of_hidden_layers
