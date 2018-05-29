import logging
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
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


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
                layer.apply(weights_initialize)

                setattr(self, 'network_{}_layer_{}'.format(network_i, i), layer)
            output_layer = nn.Linear(in_features, network_config.output_shape[0])
            setattr(self, 'layer_q_{}'.format(network_i), output_layer)


    def forward(self, input):
        q_values = []
        for network_i, network in enumerate(self.network_config.networks):
            out = input
            for i in range(len(network['layers'])):
                out = getattr(self, 'network_{}_layer_{}'.format(network_i, i))(out)
            q_values.append(getattr(self, 'layer_q_{}'.format(network_i))(out))

        return torch.stack(q_values)


class HRAModel(Model):
    """Neural Network with the HRA architecture  """

    def __init__(self, name, network_config, restore=True, learning_rate=0.001):
        self.network_config = network_config
        self.name = name

        summaries_path =  self.network_config.summaries_path + "/" + self.name
        clear_summary_path(summaries_path)
        self.summary = SummaryWriter(log_dir = summaries_path)

        model = _HRAModel(network_config)
        Model.__init__(self, model, name, network_config, restore)
        logger.info("Created network for %s " % self.name)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
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
            for i in range(len(network['layers'])):
                weight_name = 'Network{}/layer{}/weights'.format(network_i, i)
                bias_name = 'Network{}/layer{}/bias'.format(network_i, i)
                layer, _ = getattr(self.model, 'network_{}_layer_{}'.format(network_i, i))
                self.summary.add_histogram(tag = weight_name, values = layer.weight.data.clone().cpu().numpy(), global_step = steps, bins = 100000)
                self.summary.add_histogram(tag = bias_name, values = layer.bias.data.clone().cpu().numpy(), global_step = steps, bins = 100000)

            weight_name = 'Network{}/Output Layer/weights'.format(network_i, i)
            bias_name = 'Network{}/Output Layer/bias'.format(network_i, i)

            output_layer = getattr(self.model, 'layer_q_{}'.format(network_i))

            self.summary.add_histogram(tag = weight_name, values = output_layer.weight.data.clone().cpu().numpy(), global_step = steps, bins = 100000)
            self.summary.add_histogram(tag = bias_name, values = output_layer.bias.data.clone().cpu().numpy(), global_step = steps, bins = 100000)

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
            self.weights_summary(steps)
            self.summary.add_histogram(tag = "%s/Q values" % (self.name), values = combined_q_values.clone().cpu().data.numpy(), global_step = steps, bins = 100000)
            for network_i, network in enumerate(self.network_config.networks):
                 name = 'Network{}/Q_value'.format(network_i)
                 self.summary.add_histogram(tag = name, values = q_values[network_i].clone().cpu().data.numpy(), global_step = steps, bins = 100000)

        return q_actions.item(), q_values, combined_q_values


    def predict_batch(self, input):
        q_values = self.model(input)
        combined_q_values = torch.sum(q_values, 0)
        values, q_actions = torch.max(combined_q_values, 1)
        return q_actions, q_values, combined_q_values
