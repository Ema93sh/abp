from collections import OrderedDict
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .model import Model
from abp.utils import clear_summary_path
from tensorboardX import SummaryWriter

logger = logging.getLogger('root')


def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        module.bias.data.fill_(0.01)


class _DQNModel(nn.Module):
    """ Model for DQN """
    def __init__(self, network_config):
        super(_DQNModel, self).__init__()
        layers = network_config.layers
        input_shape = int(np.prod(network_config.input_shape))
        layer_modules = OrderedDict()

        for i, layer in enumerate(layers):
            layer_name = "Layer_%d" % i
            layer_modules[layer_name] = nn.Linear(input_shape, layer)
            layer_modules[layer_name + "relu"] = nn.ReLU()
            input_shape = layer

        layer_modules["OutputLayer"] = nn.Linear(input_shape, network_config.output_shape[0])
        self.layers = nn.Sequential(layer_modules)
        self.layers.apply(weights_initialize)

    def forward(self, input):
        return self.layers(input)

class DQNModel(Model):

    def __init__(self, name, network_config, use_cuda, restore = True, learning_rate = 0.001):
        self.name = name
        model = _DQNModel(network_config)
        if use_cuda:
            logger.info("Network %s is using cuda " % self.name)
            model = model.cuda()

        super(DQNModel, self).__init__(model, name, network_config, restore)
        self.network_config = network_config
        self.optimizer = Adam(self.model.parameters(), lr = self.network_config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        summaries_path =  self.network_config.summaries_path + "/" + self.name

        if not network_config.restore_network:
            clear_summary_path(summaries_path)
            self.summary = SummaryWriter(log_dir = summaries_path)
            dummy_input = torch.rand(1, int(np.prod(network_config.input_shape)))
            self.summary.add_graph(self.model, dummy_input)
        else:
            self.summary = SummaryWriter(log_dir = summaries_path)

        logger.info("Created network for %s " % self.name)

    def weights_summary(self, steps):
        for i in range(len(self.network_config.layers)):
            layer_name = "Layer_%d" % i
            weight_name = '{}/layer{}/weights'.format(self.name, i)
            bias_name = '{}/layer{}/bias'.format(self.name, i)
            weight = getattr(self.model.layers, layer_name).weight.clone().data
            bias = getattr(self.model.layers, layer_name).bias.clone().data
            self.summary.add_histogram(tag = weight_name, values = weight, global_step = steps)
            self.summary.add_histogram(tag = bias_name, values = bias, global_step = steps)

        weight_name = '{}/Output Layer/weights'.format(self.name, i)
        bias_name = '{}/Output Layer/bias'.format(self.name, i)

        weight = getattr(self.model.layers, "OutputLayer").weight.clone().data
        bias = getattr(self.model.layers, "OutputLayer").bias.clone().data

        self.summary.add_histogram(tag = weight_name, values = weight, global_step = steps)
        self.summary.add_histogram(tag = bias_name, values = bias, global_step = steps)

    def predict(self, input, steps, learning):
        q_values = self.model(input).squeeze(1)
        action = torch.argmax(q_values)

        if steps % self.network_config.summaries_step == 0 and learning:
            logger.debug("Adding network summaries!")
            self.weights_summary(steps)
            self.summary.add_histogram(tag = "%s/Q values" % (self.name), values = q_values.clone().cpu().data.numpy(), global_step = steps)

        return action.item(), q_values

    def predict_batch(self, input):
        q_values = self.model(input)
        values, q_actions = q_values.max(1)
        return q_actions, q_values

    def fit(self, q_values, target_q_values, steps):
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.summary.add_scalar(tag = "%s/Loss" % (self.name),
                                scalar_value = float(loss),
                                global_step = steps)
