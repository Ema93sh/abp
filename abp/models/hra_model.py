import logging
logger = logging.getLogger('root')

import numpy as np
import torch
import torch.nn as nn
from torch.optim import RMSprop, Adam
from tensorboardX import SummaryWriter

from .model import Model
from abp.utils import clear_summary_path


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
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        summaries_path =  self.network_config.summaries_path + "/" + self.name
        clear_summary_path(summaries_path)
        self.summary = SummaryWriter(log_dir = summaries_path)

        model = _HRAModel(network_config)
        Model.__init__(self, model, name, network_config, restore)
        logger.info("Created network for %s " % self.name)

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
                self.summary.add_histogram(weight_name, layer.weight.data.clone().numpy(), steps)
                self.summary.add_histogram(bias_name, layer.bias.data.clone().numpy(), steps)

            weight_name = 'Network{}/Output Layer/weights'.format(network_i, i)
            bias_name = 'Network{}/Output Layer/bias'.format(network_i, i)

            output_layer = getattr(self.model, 'layer_q_{}'.format(network_i))

            self.summary.add_histogram(weight_name, output_layer.weight.data.clone().numpy(), steps)
            self.summary.add_histogram(bias_name, output_layer.bias.data.clone().numpy(), steps)

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
            self.summary.add_histogram("%s/Q values" % (self.name), combined_q_values.clone().cpu().data.numpy(), steps)
            for network_i, network in enumerate(self.network_config.networks):
                 name = 'Network{}/Q_value'.format(network_i)
                 self.summary.add_histogram(name, q_values[network_i].clone().cpu().data.numpy(), steps)

        return q_actions.item(), q_values, combined_q_values


    def predict_batch(self, input):
        q_values = self.model(input)
        combined_q_values = torch.sum(q_values, 0)
        values, q_actions = torch.max(combined_q_values, 1)
        return q_actions, q_values, combined_q_values
