import os
import logging
import torch
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
        for network_i, network in enumerate(network_config.networks):
            in_features = np.prod(network_config.input_shape)
            for i, out_features in enumerate(network['layers']):
                layer = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU())
                in_features = out_features
                setattr(self, '{}_layer_{}'.format(network_i, i), layer)
            q_linear = nn.Linear(in_features, network_config.output_shape[0])
            setattr(self, '{}_layer_q'.format(network_i), q_linear)

    def forward(self, input):
        q_values = []
        for network_i, network in enumerate(self.network_config.networks):
            out = input
            for i in range(len(network['layers'])):
                out = getattr(self, '{}_layer_{}'.format(network_i, i))(out)
            q_values.append(getattr(self, '{}_layer_q'.format(network_i))(out))
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

    def fit(self, states, target, steps):
        self.optimizer.zero_grad()
        predict = self.model(states)
        loss = 0
        for i,p in enumerate(predict):
            loss += self.loss_fn(p, Variable(torch.Tensor(target[i])))
        loss.backward()
        self.optimizer.step()

    def predict(self, input):
        q_values = self.predict_batch(input.unsqueeze(0)).squeeze(axis=1)

        q_actions = np.sum(q_values,axis=0)
        return np.argmax(q_actions), q_values
