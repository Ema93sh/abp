from abp.configs.base_config import BaseConfig

default_reinforce_config = {
    "decay_steps" : 250,
    "starting_epsilon": 1.0,
    "decay_rate": 0.96,
    "discount_factor": 0.95,
    "batch_size": 35,
    "memory_size": 10000, # Total Memory size for the batch
    "summaries_path": None, # Path to store reinforcement related summaries
    "replace_frequency": 1000,
    "update_steps": 10, # Will update the model after n steps
}

class ReinforceConfig(BaseConfig):
    """Configurations related to reinfocement learning"""

    def __init__(self, config = None):
        super(ReinforceConfig, self).__init__(config, default_config = default_reinforce_config)

    #TODO Can use python feature to generate these properties? Manual Labour for now -_-


    decay_steps = property(BaseConfig.get_property("decay_steps"), BaseConfig.set_property("decay_steps"))

    starting_epsilon = property(BaseConfig.get_property("starting_epsilon"), BaseConfig.set_property("starting_epsilon"))

    decay_rate = property(BaseConfig.get_property("decay_rate"), BaseConfig.set_property("decay_rate"))

    discount_factor = property(BaseConfig.get_property("discount_factor"), BaseConfig.set_property("discount_factor"))

    batch_size = property(BaseConfig.get_property("batch_size"), BaseConfig.set_property("batch_size"))

    memory_size = property(BaseConfig.get_property("memory_size"), BaseConfig.set_property("memory_size"))

    summaries_path = property(BaseConfig.get_property("summaries_path"), BaseConfig.set_property("summaries_path"))

    replace_frequency = property(BaseConfig.get_property("replace_frequency"), BaseConfig.set_property("replace_frequency"))

    update_steps = property(BaseConfig.get_property("update_steps"), BaseConfig.set_property("update_steps"))
