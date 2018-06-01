import logging
from collections import OrderedDict
logger = logging.getLogger('root')

import numpy as np
import torch
import torch.nn as nn
from torch.optim import RMSprop, Adam
from tensorboardX import SummaryWriter

from .model import Model
from abp.utils import clear_summary_path

def weights_initialize(module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        module.bias.data.fill_(0.01)

class DecomposedModel(nn.Module):
    def __init__(self, layers, input_shape, output_shape):
        super(DecomposedModel, self).__init__()
        layer_modules = OrderedDict()

        for i, layer in enumerate(layers):
            layer_name = "Layer_%d" % i
            layer_modules[layer_name] = nn.Linear(input_shape, layer)
            layer_modules[layer_name + "relu"] = nn.ReLU()
            input_shape = layer

        layer_modules["OutputLayer"] = nn.Linear(input_shape, output_shape)
        self.layers = nn.Sequential(layer_modules)
        self.layers.apply(weights_initialize)

    def forward(self, input):
        return self.layers(input)


class _HRAModel(nn.Module):
    """ HRAModel """
    def __init__(self, network_config):
        super(_HRAModel, self).__init__()
        self.network_config = network_config
        modules = []
        for network_i, network in enumerate(network_config.networks):
            input_shape = int(np.prod(network_config.input_shape))
            model = DecomposedModel(network["layers"], input_shape, network_config.output_shape[0])
            modules.append(model)
        self.reward_models = nn.ModuleList(modules)


    def forward(self, input):
        q_values = []
        for reward_model in self.reward_models:
            q_value = reward_model(input)
            q_values.append(q_value)
        return torch.stack(q_values)


class HRAModel(Model):
    """Neural Network with the HRA architecture  """

    def __init__(self, name, network_config, restore=True):
        self.network_config = network_config
        self.name = name

        summaries_path =  self.network_config.summaries_path + "/" + self.name
        clear_summary_path(summaries_path)
        self.summary = SummaryWriter(log_dir = summaries_path)

        model = _HRAModel(network_config)
        Model.__init__(self, model, name, network_config, restore)
        logger.info("Created network for %s " % self.name)

        self.optimizer = Adam(self.model.parameters(), lr=self.network_config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        dummy_input = torch.rand(1, int(np.prod(network_config.input_shape)))
        self.summary.add_graph(self.model, dummy_input)

    def clear_weights(self, reward_type):
        for type in range(self.model.networks):
            if type != reward_type:
                getattr(self.model, 'layer_q_{}'.format(type)).weight.data.fill_(0)
                network = self.network_config.networks[type]
                for i in range(len(network['layers'])):
                    getattr(self.model, 'network_{}_layer_{}'.format(type, i)).apply(self.weights_init)

    def weights_summary(self, steps):
        for network_i, network in enumerate(self.network_config.networks):
            out = input
            model = self.model.reward_models[network_i]
            for i in range(len(network['layers'])):
                layer_name = "Layer_%d" % i
                weight_name = 'Sub Network Type:{}/layer{}/weights'.format(network_i, i)
                bias_name = 'Sub Network Type:{}/layer{}/bias'.format(network_i, i)
                weight = getattr(model.layers, layer_name).weight.clone().data
                bias = getattr(model.layers, layer_name).bias.clone().data
                self.summary.add_histogram(tag = weight_name, values = weight, global_step = steps)
                self.summary.add_histogram(tag = bias_name, values = bias, global_step = steps)

            weight_name = 'Sub Network Type:{}/Output Layer/weights'.format(network_i, i)
            bias_name = 'Sub Network Type:{}/Output Layer/bias'.format(network_i, i)

            weight = getattr(model.layers, "OutputLayer").weight.clone().data
            bias = getattr(model.layers, "OutputLayer").bias.clone().data

            self.summary.add_histogram(tag = weight_name, values = weight, global_step = steps)
            self.summary.add_histogram(tag = bias_name, values = bias, global_step = steps)

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

    def fit(self, q_values, target_q_values, steps):
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.summary.add_scalar(tag = "%s/Loss" % (self.name),
                                scalar_value = float(loss),
                                global_step = steps)

    def predict(self, input, steps):
        q_values = self.model(input).squeeze(1)
        combined_q_values = torch.sum(q_values, 0)
        values, q_actions = torch.max(combined_q_values, 0)

        if steps % self.network_config.summaries_step == 0:
            logger.debug("Adding network summaries!")
            self.weights_summary(steps)
            self.summary.add_histogram(tag = "%s/Q values" % (self.name), values = combined_q_values.clone().cpu().data.numpy(), global_step = steps)
            for network_i, network in enumerate(self.network_config.networks):
                 name = 'Sub Network Type:{}/Q_value'.format(network_i)
                 self.summary.add_histogram(tag = name, values = q_values[network_i].clone().cpu().data.numpy(), global_step = steps)

        return q_actions.item(), q_values, combined_q_values


    def predict_batch(self, input):
        q_values = self.model(input)
        combined_q_values = torch.sum(q_values, 0)
        values, q_actions = torch.max(combined_q_values, 1)
        return q_actions, q_values, combined_q_values
