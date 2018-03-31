import os
import logging
import torch
import copy
import torch.nn as nn
from torch.optim import RMSprop
import numpy as np
from .model import Model
from torch.autograd import Variable
import numpy as np

logger = logging.getLogger('root')

class _HRAModel(nn.Module):
    def __init__(self, network_config):
        super(_HRAModel, self).__init__()
        self.network_config = network_config
        self.networks = len(network_config.networks)
        for network_i, network in enumerate(network_config.networks):
            in_features = int(np.prod(network_config.input_shape))
            for i, out_features in enumerate(network['layers']):
                layer = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU())
                in_features = out_features
                setattr(self, '{}_layer_{}'.format(network_i, i), layer)
            q_linear = nn.Linear(in_features, network_config.output_shape[0])
            setattr(self, 'layer_q_{}'.format(network_i), q_linear)

    def forward(self, input):
        q_values = []
        input = input.view((input.shape[0], int(np.prod(input.shape[1:]))))
        for network_i, network in enumerate(self.network_config.networks):
            out = input
            for i in range(len(network['layers'])):
                out = getattr(self, '{}_layer_{}'.format(network_i, i))(out)
            q_values.append(getattr(self, 'layer_q_{}'.format(network_i))(out))
        return q_values


class HRAModel(Model):
    """Neural Network with the HRA architecture  """

    def __init__(self, name, network_config, restore=True, learning_rate=0.001):
        logger.info("Building network for %s" % name)
        self.network_config = network_config
        model = _HRAModel(network_config)
        Model.__init__(self, model, name, network_config, restore)
        logger.info("Created network for %s " % self.name)
        self.optimizer = RMSprop(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.weights = {}

    def clear_weights(self, reward_type):
        for type in range(self.model.networks):
            if type != reward_type:
                self.weights[reward_type] = getattr(self.model, 'layer_q_{}'.format(reward_type)).weight.data
                getattr(self.model, 'layer_q_{}'.format(reward_type)).weight.data.fill_(0)

    def restore_weights(self):
        for reward_type, weights in self.weights.items():
            getattr(self.model, 'layer_q_{}'.format(reward_type)).weight.data = weights
        self.weights = {}

    def fit(self, states, target, steps):
        self.optimizer.zero_grad()
        predict = self.model(states)
        loss = 0
        for i, p in enumerate(predict):
            loss += self.loss_fn(p, Variable(torch.Tensor(target[i])))
        loss.backward()
        self.optimizer.step()

    def predict(self, input):
        q_values = self.predict_batch(input.unsqueeze(0)).squeeze(axis=1)
        q_actions = np.sum(q_values, axis=0)
        return np.argmax(q_actions), q_values
