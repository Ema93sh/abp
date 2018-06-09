from abp.configs.base_config import BaseConfig

default_evaluation_config = {
    "name": "default name",
    "env": None,  # The open AI gym environment
    "summaries_path": None,
    "training_episodes": 100,
    "test_episodes": 100,
    "render": False,
    "contrastive": False,
    "sleep": 1
}


class EvaluationConfig(BaseConfig):
    """ Everything related to evaluations """

    def __init__(self, config=None):
        super(EvaluationConfig, self).__init__(
            config=config, default_config=default_evaluation_config)

    name = property(BaseConfig.get_property("name"),
                    BaseConfig.set_property("name"))

    env = property(BaseConfig.get_property("env"),
                   BaseConfig.set_property("env"))

    summaries_path = property(BaseConfig.get_property("summaries_path"),
                              BaseConfig.set_property("summaries_path"))

    training_episodes = property(BaseConfig.get_property("training_episodes"),
                                 BaseConfig.set_property("training_episodes"))

    test_episodes = property(BaseConfig.get_property("test_episodes"),
                             BaseConfig.set_property("test_episodes"))

    render = property(BaseConfig.get_property("render"),
                      BaseConfig.set_property("render"))

    contrastive = property(BaseConfig.get_property("contrastive"),
                           BaseConfig.set_property("contrastive"))

    sleep = property(BaseConfig.get_property("sleep"),
                     BaseConfig.set_property("sleep"))
