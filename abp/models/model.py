import os
import torch
import logging
import numpy as np
logger = logging.getLogger('root')


def ensure_directory_exits(directory_path):
    """creates directory if path doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


class Model(object):
    def __init__(self, model, name, network_config, restore=True):
        self.restore = restore
        self.name = name.replace(" ", "_")
        self.network_config = network_config
        file_path = ensure_directory_exits(self.network_config.network_path)
        self.model_path = os.path.join(file_path, self.name + '.p')
        self.model = model
        self.restore_network()

    def save_network(self):
        if self.network_config.network_path and self.network_config.save_network:
            logger.info("Saving network for..." + self.name)
            logger.info("Saving the network at %s" % self.model_path)
            torch.save(self.model.state_dict(), self.model_path)

    def restore_network(self):
        if (self.restore and
            self.network_config.restore_network and
                self.network_config.network_path):
            if os.path.exists(self.model_path):
                logger.info("Restoring network for %s " % self.name)
                self.model.load_state_dict(torch.load(self.model_path))
            else:
                logger.info("Model does not exist %s" % self.model_path)

    def predict_batch(self, input):
        return np.array([x for x in self.model(input)])

    def predict(self, input):
        raise NotImplementedError('Needs to be implemented!')

    def fit(self):
        raise NotImplementedError('Needs to be implemented!')

    def replace(self, dest):
        self.model.load_state_dict(dest.model.state_dict())
