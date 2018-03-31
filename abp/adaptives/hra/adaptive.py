import logging

logger = logging.getLogger('root')

from copy import deepcopy
import numpy as np
import torch
from torch.autograd import Variable

from abp.adaptives.common.memory import Memory
from abp.adaptives.common.experience import Experience
from abp.utils import clear_summary_path
from abp.models import HRAModel
from tensorboardX import SummaryWriter
import excitationbp as eb

# TODO Too many duplicate code. Need to refactor!

class HRAAdaptive(object):
    """HRAAdaptive using HRA architecture"""

    def __init__(self, name, choices, reward_types, network_config, reinforce_config, log=True):
        super(HRAAdaptive, self).__init__()
        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.update_frequency = reinforce_config.update_frequency

        self.replay_memory = Memory(self.reinforce_config.memory_size)
        self.learning = True
        self.explanation = False

        self.steps = 0
        self.previous_state = None
        self.previous_action = None
        self.reward_types = reward_types

        self.clear_rewards()


        self.total_reward = 0

        self.eval_model = HRAModel(self.name + "_eval", self.network_config)
        self.target_model = HRAModel(self.name + "_target", self.network_config)
        self.log = log
        if self.log:
            self.summary = SummaryWriter()
        self.episode = 0

    def __del__(self):
        self.summary.close()

    def should_explore(self):
        epsilon = np.max([0.1, self.reinforce_config.starting_epsilon * (
                    self.reinforce_config.decay_rate ** (self.steps / self.reinforce_config.decay_steps))])
        if self.log:
            self.summary.add_scalar(tag='epsilon', scalar_value=epsilon, global_step=self.steps)
        return np.random.choice([True, False], p=[epsilon, 1 - epsilon])

    def predict(self, state):
        self.steps += 1
        saliencies = None

        # add to experience
        if self.previous_state is not None and self.previous_action is not None:
            experience = Experience(self.previous_state, self.previous_action, self.reward_list(), state)
            self.replay_memory.add(experience)

        if self.learning and self.should_explore():
            action = np.random.choice(len(self.choices))
            q_values = [None] * len(self.choices)  # TODO should it be output shape or from choices?
            choice = self.choices[action]
        else:
            _state = Variable(torch.Tensor(state)).unsqueeze(0)
            action, q_values = self.eval_model.predict(_state)

            choice = self.choices[action]

        if self.learning and self.steps % self.update_frequency == 0:
            logger.debug("Replacing target model for %s" % self.name)
            self.target_model.replace(self.eval_model)

        if self.explanation:
            saliencies = self.generate_saliencies(_state)


        self.update()

        self.clear_rewards()

        self.previous_state = state
        self.previous_action = action

        return choice, q_values, saliencies

    def disable_learning(self):
        logger.info("Disabled Learning for %s agent" % self.name)
        self.eval_model.save_network()
        self.target_model.save_network()

        self.learning = False
        self.episode = 0

    def enable_explanation(self):
        self.explanation = True

    def disable_explanation(self):
        self.explanation = False

    def generate_saliencies(self, state):
        eb.use_eb(True)

        prob_outputs = Variable(torch.zeros((len(self.choices),)))
        saliencies = {}
        for idx, choice in enumerate(self.choices):
            choice_saliencies = {}
            prob_outputs[idx] = 1

            for reward_idx, reward_type in enumerate(self.reward_types):
                explainable_model = HRAModel(self.name + "_explain", self.network_config, restore = False)
                explainable_model.replace(self.eval_model)

                explainable_model.clear_weights(reward_idx)
                saliency = eb.excitation_backprop(explainable_model.model, state, prob_outputs, contrastive = False, target_layer = 0)

                choice_saliencies[reward_type] = np.squeeze(saliency.view(*state.shape).data.numpy())

            # for overall reward
            saliency = eb.excitation_backprop(self.eval_model.model, state, prob_outputs, contrastive = False, target_layer = 0)
            choice_saliencies["all"] = np.squeeze(saliency.view(*state.shape).data.numpy())

            saliencies[choice] = choice_saliencies

        eb.use_eb(False)
        return saliencies

    def end_episode(self, state):
        if not self.learning:
            return

        logger.info("End of Episode %d with total reward %d" % (self.episode + 1, self.total_reward))

        self.episode += 1
        print('episode:', self.episode)
        if self.log:
            self.summary.add_scalar(tag='%s agent reward' % self.name,scalar_value=self.total_reward,
                                    global_step=self.episode)
            print('agent reward:', self.total_reward)

        experience = Experience(self.previous_state, self.previous_action, self.reward_list(), state, is_terminal=True)
        self.replay_memory.add(experience)

        self.clear_rewards()
        self.total_reward = 0

        self.previous_state = None
        self.previous_action = None

        self.update()

    def reward_list(self):
        reward = [0] * len(self.reward_types)

        for i, reward_type in enumerate(sorted(self.reward_types)):
            reward[i] = self.current_reward[reward_type]

        return reward

    def clear_rewards(self):
        self.current_reward = {}
        for reward_type in self.reward_types:
            self.current_reward[reward_type] = 0


    def reward(self, reward_type, value):
        self.current_reward[reward_type] += value
        self.total_reward += value


    def update(self):
        if self.replay_memory.current_size < self.reinforce_config.batch_size:
            return

        batch = self.replay_memory.sample(self.reinforce_config.batch_size)

        # TODO: Convert to tensor operations instead of for loops

        states = [experience.state for experience in batch]

        next_states = [experience.next_state for experience in batch]
        states = Variable(torch.Tensor(states))
        next_states = Variable(torch.Tensor(next_states))

        is_terminal = [0 if experience.is_terminal else 1 for experience in batch]

        actions = [experience.action for experience in batch]

        reward = np.array([experience.reward for experience in batch])

        q_next = self.target_model.predict_batch(next_states)

        q_2 = np.mean(q_next, axis = 2)

        q_2 = is_terminal * q_2

        q_values = self.eval_model.predict_batch(states)

        q_target = q_values.copy()

        batch_index = np.arange(self.reinforce_config.batch_size, dtype=np.int32)

        q_target[:, batch_index, actions] = np.transpose(reward) + self.reinforce_config.discount_factor * q_2

        self.eval_model.fit(states, q_target, self.steps)
