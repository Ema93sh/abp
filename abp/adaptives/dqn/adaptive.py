import logging
import numpy as np
import torch
from torch.autograd import Variable
from abp.adaptives.common.memory import Memory
from abp.adaptives.common.experience import Experience
from abp.models import DQNModel
from tensorboardX import SummaryWriter

logger = logging.getLogger('root')


class DQNAdaptive(object):
    """Adaptive which uses the  DQN algorithm"""

    def __init__(self, name, choices, network_config, reinforce_config, log=True):
        super(DQNAdaptive, self).__init__()
        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.update_frequency = reinforce_config.update_frequency

        self.replay_memory = Memory(self.reinforce_config.memory_size)
        self.learning = True

        self.steps = 0
        self.previous_state = None
        self.previous_action = None
        self.current_reward = 0
        self.total_reward = 0
        self.log = log
        if self.log:
            self.summary = SummaryWriter()

        self.target_model = DQNModel(self.name + "_target", self.network_config)
        self.eval_model = DQNModel(self.name + "_eval", self.network_config)

        self.episode = 0

    def __del__(self):
        pass

    def should_explore(self):
        epsilon = np.max([0.1, self.reinforce_config.starting_epsilon * (
                self.reinforce_config.decay_rate ** (self.steps / self.reinforce_config.decay_steps))])
        if self.log:
            self.summary.add_scalar(tag='epsilon', scalar_value=epsilon, global_step=self.steps)
        return np.random.choice([True, False], p=[epsilon, 1 - epsilon])

    def predict(self, state):
        self.steps += 1

        # add to experience
        if self.previous_state is not None:
            experience = Experience(self.previous_state, self.previous_action, self.current_reward, state)
            self.replay_memory.add(experience)

        if self.learning and self.should_explore():
            action = np.random.choice(len(self.choices))
            q_values = [None] * len(self.choices)  # TODO should it be output shape or from choices?
            choice = self.choices[action]
        else:
            _state = Variable(torch.Tensor(state)).unsqueeze(0)
            q_values = self.eval_model.predict(_state)
            q_values = q_values.data.numpy()[0]
            action = np.argmax(q_values)
            choice = self.choices[action]

        if self.learning and self.steps % self.update_frequency == 0:
            logger.debug("Replacing target model for %s" % self.name)
            self.target_model.replace(self.eval_model)

        self.update()

        self.current_reward = 0

        self.previous_state = state
        self.previous_action = action

        return choice, q_values

    def disable_learning(self):
        logger.info("Disabled Learning for %s agent" % self.name)
        self.eval_model.save_network()
        self.target_model.save_network()

        self.learning = False
        self.episode = 0

    def end_episode(self, state):
        if not self.learning:
            return

        if self.episode % 100 == 0:
            logger.info("End of Episode %d with total reward %d" % (self.episode + 1, self.total_reward))

        self.episode += 1
        if self.log:
            self.summary.add_scalar(tag='%s agent reward' % self.name,scalar_value=self.total_reward, global_step=self.episode)
        experience = Experience(self.previous_state, self.previous_action, self.current_reward, state, True)
        self.replay_memory.add(experience)

        self.current_reward = 0
        self.total_reward = 0

        self.previous_state = None
        self.previous_action = None

        if self.replay_memory.current_size > 30:
            self.update()

    def reward(self, r):
        self.total_reward += r
        self.current_reward += r

    def update(self):
        if self.replay_memory.current_size < self.reinforce_config.batch_size:
            return

        batch = self.replay_memory.sample(self.reinforce_config.batch_size)

        states = [experience.state for experience in batch]
        next_states = [experience.next_state for experience in batch]

        states = Variable(torch.Tensor(states))
        next_states = Variable(torch.Tensor(next_states))

        is_terminal = [0 if experience.is_terminal else 1 for experience in batch]

        actions = [experience.action for experience in batch]
        reward = [experience.reward for experience in batch]

        q_next = self.target_model.predict(next_states)
        q_max = torch.max(q_next, dim=1)[0].data.numpy()
        q_max = np.array([a * b if a == 0 else b for a, b in zip(is_terminal, q_max)])

        q_predict = self.eval_model.predict(states)

        q_target = q_predict.data.numpy()
        batch_index = np.arange(self.reinforce_config.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = reward + self.reinforce_config.discount_factor * q_max
        q_target = Variable(torch.Tensor(q_target))
        self.eval_model.fit(states, q_target, self.steps)
