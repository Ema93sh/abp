import logging
import torch.nn as nn
from torch.optim import RMSprop
from .model import Model
import numpy as np

logger = logging.getLogger('root')


class _DQNModel(nn.Module):
    """Neural Network for the DQN algorithm """

    def __init__(self, network_config):
        super(_DQNModel, self).__init__()
        self.num_layers = len(network_config.layers)
        in_features = np.prod(network_config.input_shape)
        for i, out_features in enumerate(network_config.layers):
            layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU())
            in_features = out_features
            setattr(self, 'layer_{}'.format(i), layer)
        self.q_linear = nn.Linear(in_features, network_config.output_shape[0])

    def forward(self, input):
        out = input.view((input.shape[0], np.prod(input.shape[1:])))
        for i in range(self.num_layers):
            out = getattr(self, 'layer_{}'.format(i))(out)
        return self.q_linear(out)


class DQNModel(Model):

    def __init__(self, name, network_config, restore=True, learning_rate=0.001):
        logger.info("Building network for %s" % name)
        self.network_config = network_config
        model = _DQNModel(network_config)
        Model.__init__(self, model, name, network_config, restore)
        logger.info("Created network for %s " % self.name)
        self.optimizer = RMSprop(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def fit(self, states, target, steps):
        self.optimizer.zero_grad()
        predict = self.model(states)
        loss = self.loss_fn(predict, target)
        loss.backward()
        self.optimizer.step()

    def predict(self, input):
        return self.model(input)
