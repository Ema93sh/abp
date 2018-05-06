import logging
logger = logging.getLogger('root')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from abp.utils import clear_summary_path
from .model import Model

class _ActorModel(nn.Module):
    def __init__(self, network_config):
        self.network_config = network_config
        in_features = int(np.prod(network_config.input_shape))
        for i, out_features in enumerate(network_config['layers']):
            layer = nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.ReLU())
            in_features = out_features
            setattr(self, 'layer_{}'.format(i), layer)
        self.action_layer = nn.Linear(in_features, network_config.output_shape[0])
        pass

    def forward(self, input):
        out = input
        for i, out_features in enumerate(self.network_config['layers']):
            out = getattr(self, 'layer_{}'.format(i), layer)(out)
        out = self.action_layer(out)
        out = F.sigmoid(out)
        return out


class ActorModel(Model):
    """ A model for actor (policy network)  """

    def __init__(self,  name, network_config, restore = True, learning_rate = 0.001):
        self.network_config = network_config
        logger.info("Building network for %s" % name)
        model = _ActorModel(network_config)
        Model.__init__(self, model, name, network_config, restore)
        logger.info("Created network for %s" % self.name)
        self.optimizer = RMSprop(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def predict(self, state):
        return self.model(state)


    def predict_batch(self, batch):
        return [self.model(state) for state in batch]


    def fit(self, policy_loss):
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        pass
