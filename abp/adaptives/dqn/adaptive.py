import logging
logger = logging.getLogger('root')

import time
import random
import pickle
import os

import torch
from tensorboardX import SummaryWriter
from baselines.common.schedules import LinearSchedule

from abp.utils import clear_summary_path
from abp.models import DQNModel
from abp.adaptives.common.prioritized_memory.memory import PrioritizedReplayBuffer

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



class DQNAdaptive(object):
    """Adaptive which uses the  DQN algorithm"""

    def __init__(self, name, choices, network_config, reinforce_config):
        super(DQNAdaptive, self).__init__()
        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config

        self.replay_memory = PrioritizedReplayBuffer(self.reinforce_config.memory_size, 0.6)
        self.learning = True

        #Global
        self.steps = 0
        self.reward_history = []
        self.best_reward_mean = 0
        self.episode = 0

        self.reset()

        if not self.network_config.restore_network:
            clear_summary_path(self.reinforce_config.summaries_path + "/" + self.name)
        else:
            self.restore_state()

        self.summary = SummaryWriter(log_dir = self.reinforce_config.summaries_path + "/" + self.name)

        self.target_model = DQNModel(self.name + "_target", self.network_config, use_cuda)
        self.eval_model = DQNModel(self.name + "_eval", self.network_config, use_cuda)

        self.beta_schedule = LinearSchedule(self.reinforce_config.beta_timesteps, initial_p = self.reinforce_config.beta_initial, final_p = self.reinforce_config.beta_final)

    def __del__(self):
        self.save()
        self.summary.close()


    def should_explore(self):
        self.epsilon = max([0.1, self.reinforce_config.starting_epsilon * (self.reinforce_config.decay_rate ** (self.steps / self.reinforce_config.decay_steps))])
        self.summary.add_scalar(tag='epsilon', scalar_value=self.epsilon, global_step=self.steps)

        return random.random() < self.epsilon

    def predict(self, state):
        self.steps += 1

        # add to experience
        if self.previous_state is not None:
            experience = (self.previous_state, self.previous_action, self.current_reward, state, False)
            self.replay_memory.add(*experience)

        if self.learning and self.should_explore():
            q_values = None
            choice = random.choice(self.choices)
            action = self.choices.index(choice)
        else:
            self.prediction_time -= time.time()
            _state = Tensor(state).unsqueeze(0)
            action, q_values = self.eval_model.predict(_state, self.steps, self.learning)
            choice = self.choices[action]
            self.prediction_time += time.time()

        if self.learning and self.steps % self.reinforce_config.replace_frequency == 0:
            logger.debug("Replacing target model for %s" % self.name)
            self.target_model.replace(self.eval_model)

        if self.learning and self.steps % self.reinforce_config.update_steps == 0:
            self.update_time -= time.time()
            self.update()
            self.update_time += time.time()

        self.current_reward = 0
        self.previous_state = state
        self.previous_action = action

        return choice, q_values


    def disable_learning(self):
        logger.info("Disabled Learning for %s agent" % self.name)
        self.save()
        self.learning = False
        self.episode = 0


    def end_episode(self, state):
        if not self.learning:
            return

        episode_time = time.time() - self.episode_time

        self.reward_history.append(self.total_reward)

        logger.info("End of Episode %d with total reward %.2f, epsilon %.2f" % (self.episode + 1, self.total_reward, self.epsilon))
        logger.debug("Episode Time: %.2fs Prediction Time: %.2f Update Time %.2f" % (episode_time, self.prediction_time, self.update_time))

        self.episode += 1
        self.summary.add_scalar(tag = '%s/Episode Reward' % self.name,
                                scalar_value = self.total_reward,
                                global_step = self.episode)

        experience = (self.previous_state, self.previous_action, self.current_reward, state, True)
        self.replay_memory.add(*experience)
        self.save()
        self.reset()


    def reset(self):
        self.episode_time = time.time()
        self.current_reward = 0
        self.total_reward = 0
        self.previous_state = None
        self.previous_action = None
        self.prediction_time = 0
        self.update_time = 0

    def restore_state(self):
        restore_path = self.network_config.network_path + "/adaptive.info"
        if self.network_config.network_path and os.path.exists(restore_path):
            logger.info("Restoring state from %s" % self.network_config.network_path)
            with open(restore_path, "rb") as file:
                info = pickle.load(file)

            self.steps = info["steps"]
            self.best_reward_mean = info["best_reward_mean"]
            self.episode = info["episode"]

    def save(self, force = False):
        info = {
            "steps": self.steps,
            "best_reward_mean": self.best_reward_mean,
            "episode": self.episode
        }

        if force:
            logger.info("Forced to save network")
            self.eval_model.save_network()
            self.target_model.save_network()
            pickle.dump(info, self.network_config.network_path + "adaptive.info")


        if len(self.reward_history) >= self.network_config.save_steps and self.episode % self.network_config.save_steps == 0:
            current_reward_mean = sum(self.reward_history[-self.network_config.save_steps:]) /  (self.network_config.save_steps * 1.0)
            if current_reward_mean >= self.best_reward_mean:
                self.best_reward_mean = current_reward_mean
                logger.info("Saving network. Found new best reward (%.2f)" % current_reward_mean)
                self.eval_model.save_network()
                self.target_model.save_network()
                with open(self.network_config.network_path + "/adaptive.info", "wb") as file:
                    pickle.dump(info, file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                logger.info("The best reward is still %.2f. Not saving" % self.best_reward_mean)


    def reward(self, r):
        self.total_reward += r
        self.current_reward += r


    def update(self):
        if self.steps <= self.reinforce_config.batch_size:
            return

        beta = self.beta_schedule.value(self.steps)
        self.summary.add_scalar(tag='%s/Beta' % self.name, scalar_value=beta, global_step=self.steps)

        states, actions, reward, next_states, is_terminal, weights, batch_idxes = self.replay_memory.sample(self.reinforce_config.batch_size, beta)

        self.summary.add_histogram(tag = '%s/Batch Indices' % self.name,
                                   values = Tensor(batch_idxes),
                                   global_step = self.steps)

        states = FloatTensor(states)
        next_states = FloatTensor(next_states)
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.reinforce_config.batch_size, dtype = torch.long)

        #Current Q Values
        q_actions, q_values  = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]


        #Calculate target
        actions, q_next = self.target_model.predict_batch(next_states)
        q_max = q_next.max(1)[0].detach()
        q_max = (1 - terminal) * q_max

        q_target = reward + self.reinforce_config.discount_factor * q_max


        #update model
        self.eval_model.fit(q_values, q_target, self.steps)

        #Update priorities
        td_errors = q_values - q_target
        new_priorities = torch.abs(td_errors) + 1e-6 #prioritized_replay_eps
        self.replay_memory.update_priorities(batch_idxes, new_priorities.data)
