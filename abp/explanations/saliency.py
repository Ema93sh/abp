import copy
import os

import numpy as np
import torch
import torchvision

from abp.models import HRAModel

from saliency import SaliencyMethod, MapType, generate_saliency

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

    def generate_saliencies(self, step, state, choice_descriptions, layer_names, file_path_prefix = "saved_saliencies/", reshape = None):
        state = Tensor(state).unsqueeze(0)

        file_path_prefix = file_path_prefix + "step_" + str(step) + "/"
        for saliency_method in SaliencyMethod:
            file_path  = file_path_prefix + str(saliency_method) + "/"
            for idx, choice in enumerate(self.adaptive.choices):
                choice_saliency = {}
                self.adaptive.eval_model.model.combined = True
                saliencies = self.generate_saliency_for(state, [idx], saliency_method)
                choice_saliency["all"] = saliencies[MapType.ORIGINAL]
                self.save_saliencies(saliencies, file_path + "choice_" + str(choice_descriptions[idx]) + "/combined/", reshape, layer_names)
                self.adaptive.eval_model.model.combined = False

                for reward_idx, reward_type in enumerate(self.adaptive.reward_types):
                    saliencies = self.generate_saliency_for(state, [idx], saliency_method, reward_idx)
                    self.save_saliencies(saliencies, file_path + "choice_" + str(choice_descriptions[idx]) + "/" + "reward_type_" + str(reward_type) + "/", reshape, layer_names)
                    choice_saliency[reward_type] = saliencies[MapType.ORIGINAL]
                saliencies[choice] = choice_saliency


    def save_saliencies(self, saliencies, file_path_prefix, reshape, layer_names):
        for map_type, saliency in saliencies.items():
            saliency = saliency.view(*reshape)
            for idx, layer_name in enumerate(layer_names):
                if not os.path.exists(file_path_prefix + str(map_type)):
                        os.makedirs(file_path_prefix + str(map_type))
                torchvision.utils.save_image(saliency[:, :, idx],  file_path_prefix + str(map_type) + "/" + layer_name + ".png", normalize=True)


    def generate_saliency_for(self, state, choice, saliency_method, reward_idx = None):
        model = self.adaptive.eval_model.model

        if reward_idx is not None:
            model = self.adaptive.eval_model.get_model_for(reward_idx)

        if type(choice) == int:
            choice = [choice]

        return generate_saliency(model, state, choice, type = saliency_method)
