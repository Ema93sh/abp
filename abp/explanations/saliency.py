import copy

import numpy as np
import torch

from abp.models import HRAModel

import excitationbp as eb

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class Saliency(object):
    """ Saliency for an Adaptive Variable """
    #TODO currenly supports only HRA Adaptive

    def __init__(self, adaptive):
        super(Saliency, self).__init__()
        self.adaptive = adaptive

    def generate_saliencies(self, state, contrastive = False):
        eb.use_eb(True)

        state = Tensor(state).unsqueeze(0)

        saliencies = {}
        for idx, choice in enumerate(self.adaptive.choices):
            choice_saliencies = {}
            prob_outputs = torch.zeros((len(self.adaptive.choices),))
            prob_outputs[idx] = 1

            for reward_idx, reward_type in enumerate(self.adaptive.reward_types):
                explainable_model = HRAModel(self.adaptive.name + "_explain", self.adaptive.network_config, False, restore = False)
                explainable_model.replace(self.adaptive.eval_model)

                explainable_model.clear_weights(reward_idx)

                layer_top = explainable_model.top_layer(reward_idx)

                saliency = eb.excitation_backprop(explainable_model.model, state, prob_outputs, contrastive = contrastive, layer_top = layer_top, target_layer = 0)

                choice_saliencies[reward_type] = np.squeeze(saliency.view(*state.shape).data.numpy())

            # for overall reward
            saliency = eb.excitation_backprop(self.adaptive.eval_model.model, state, prob_outputs, contrastive = contrastive, target_layer = 0)
            choice_saliencies["all"] = np.squeeze(saliency.view(*state.shape).data.numpy())

            saliencies[choice] = choice_saliencies

        eb.use_eb(False)
        return saliencies
