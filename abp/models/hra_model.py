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
                setattr(self, 'network_{}_layer_{}'.format(network_i, i), layer)
            q_linear = nn.Linear(in_features, network_config.output_shape[0])
            setattr(self, 'layer_q_{}'.format(network_i), q_linear)

    def forward(self, input):
        q_values = []
        input = input.view((input.shape[0], int(np.prod(input.shape[1:]))))
        for network_i, network in enumerate(self.network_config.networks):
            out = input
            for i in range(len(network['layers'])):
                out = getattr(self, 'network_{}_layer_{}'.format(network_i, i))(out)
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

    def clear_weights(self, reward_type):
        for type in range(self.model.networks):
            if type != reward_type:
                getattr(self.model, 'layer_q_{}'.format(type)).weight.data.fill_(0)
                network = self.network_config.networks[type]
                for i in range(len(network['layers'])):
                    getattr(self.model, 'network_{}_layer_{}'.format(type, i)).apply(self.weights_init)

    def display_weights(self):
        for network_i, network in enumerate(self.network_config.networks):
            out = input
            for i in range(len(network['layers'])):
                print('*****************network_{}_layer_{}'.format(network_i, i))
                l, _ = getattr(self.model, 'network_{}_layer_{}'.format(network_i, i))
                print(l.weight.data)
                print('-----------------network_{}_layer_{}'.format(network_i, i))

            print('*************layer_q_{}'.format(network_i))
            print(getattr(self.model, 'layer_q_{}'.format(network_i)).weight.data)
            print('-------------layer_q_{}'.format(network_i))

    def top_layer(self, reward_type):
        return getattr(self.model, 'layer_q_{}'.format(reward_type))

    def weights_init(self, m):
        classname = m.__class__.__name__
        if type(m) == nn.Linear:
            m.weight.data.fill_(0)
            m.bias.data.fill_(0)
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.0, 0.0)
            m.bias.data.fill_(0)

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
