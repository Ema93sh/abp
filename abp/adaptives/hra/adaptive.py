import logging
import time
import random
import pickle
import os
import sys

import torch
from baselines.common.schedules import LinearSchedule


from abp.adaptives.common.prioritized_memory.memory import PrioritizedReplayBuffer
from abp.utils import clear_summary_path
from abp.models import HRAModel
from tensorboardX import SummaryWriter


logger = logging.getLogger('root')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class HRAAdaptive(object):
    """HRAAdaptive using HRA architecture"""

    def __init__(self, name, choices, reward_types, network_config, reinforce_config):
        super(HRAAdaptive, self).__init__()
        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.replace_frequency = reinforce_config.replace_frequency
        self.memory = PrioritizedReplayBuffer(self.reinforce_config.memory_size, 0.6)
        self.learning = True
        self.reward_types = reward_types
        self.steps = 0
        self.episode = 0
        self.reward_history = []
        self.best_reward_mean = -sys.maxsize

        self.beta_schedule = LinearSchedule(self.reinforce_config.beta_timesteps,
                                            initial_p=self.reinforce_config.beta_initial,
                                            final_p=self.reinforce_config.beta_final)

        self.epsilon_schedule = LinearSchedule(self.reinforce_config.epsilon_timesteps,
                                               initial_p=self.reinforce_config.starting_epsilon,
                                               final_p=self.reinforce_config.final_epsilon)

        self.reset()

        self.eval_model = HRAModel(self.name + "_eval", self.network_config, use_cuda)
        self.target_model = HRAModel(self.name + "_target", self.network_config, use_cuda)

        reinforce_summary_path = self.reinforce_config.summaries_path + "/" + self.name

        if not network_config.restore_network:
            clear_summary_path(reinforce_summary_path)
        else:
            self.restore_state()

        self.summary = SummaryWriter(log_dir=reinforce_summary_path)

    def __del__(self):
        self.save()
        self.summary.close()

    def should_explore(self):
        self.epsilon = self.epsilon_schedule.value(self.steps)
        self.summary.add_scalar(tag='%s/Epsilon' % self.name,
                                scalar_value=self.epsilon,
                                global_step=self.steps)

        return random.random() < self.epsilon

    def predict(self, state):
        self.steps += 1

        if (self.previous_state is not None and
                self.previous_action is not None):
            self.memory.add(self.previous_state,
                            self.previous_action,
                            self.reward_list(),
                            state, 0)

        if self.learning and self.should_explore():
            action = random.choice(list(range(len(self.choices))))
            q_values = None
            combined_q_values = None
            choice = self.choices[action]
        else:
            _state = Tensor(state).unsqueeze(0)
            model_start_time = time.time()
            action, q_values, combined_q_values = self.eval_model.predict(_state,
                                                                          self.steps,
                                                                          self.learning)
            choice = self.choices[action]
            self.model_time += time.time() - model_start_time

        if self.learning and self.steps % self.replace_frequency == 0:
            logger.debug("Replacing target model for %s" % self.name)
            self.target_model.replace(self.eval_model)

        if (self.learning and
                self.steps > self.reinforce_config.update_start and
                self.steps % self.reinforce_config.update_steps == 0):

            update_start_time = time.time()
            self.update()
            self.update_time += time.time() - update_start_time

        self.clear_current_rewards()

        self.previous_state = state
        self.previous_action = action

        return choice, q_values, combined_q_values

    def disable_learning(self):
        logger.info("Disabled Learning for %s agent" % self.name)
        self.save()

        self.learning = False
        self.episode = 0

    def end_episode(self, state):
        if not self.learning:
            return

        self.reward_history.append(self.total_reward)

        logger.info("End of Episode %d with total reward %.2f, epsilon %.2f" %
                    (self.episode + 1, self.total_reward, self.epsilon))

        self.episode += 1
        self.summary.add_scalar(tag='%s/Episode Reward' % self.name,
                                scalar_value=self.total_reward,
                                global_step=self.episode)

        for reward_type in self.reward_types:
            tag = '%s/Decomposed Reward/%s' % (self.name, reward_type)
            value = self.decomposed_total_reward[reward_type]
            self.summary.add_scalar(tag=tag, scalar_value=value, global_step=self.episode)

        self.memory.add(self.previous_state,
                        self.previous_action,
                        self.reward_list(),
                        state, 1)

        self.episode_time = time.time() - self.episode_time

        logger.debug("Episode Time: %.2f, "
                     "Model prediction time: %.2f, "
                     "Updated time: %.2f, "
                     "Update fit time: %.2f" % (self.episode_time,
                                                self.model_time,
                                                self.update_time,
                                                self.fit_time))

        self.save()
        self.reset()

    def reset(self):
        self.clear_current_rewards()
        self.clear_episode_rewards()

        self.previous_state = None
        self.previous_action = None
        self.episode_time = time.time()
        self.update_time = 0
        self.fit_time = 0
        self.model_time = 0

    def reward_list(self):
        reward = [0] * len(self.reward_types)

        for i, reward_type in enumerate(sorted(self.reward_types)):
            reward[i] = self.current_reward[reward_type]

        return reward

    def clear_current_rewards(self):
        self.current_reward = {}
        for reward_type in self.reward_types:
            self.current_reward[reward_type] = 0

    def clear_episode_rewards(self):
        self.total_reward = 0
        self.decomposed_total_reward = {}
        for reward_type in self.reward_types:
            self.decomposed_total_reward[reward_type] = 0

    def reward(self, reward_type, value):
        self.current_reward[reward_type] += value
        self.decomposed_total_reward[reward_type] += value
        self.total_reward += value

    def restore_state(self):
        restore_path = self.network_config.network_path + "/adaptive.info"

        if self.network_config.network_path and os.path.exists(restore_path):
            logger.info("Restoring state from %s" % self.network_config.network_path)

            with open(restore_path, "rb") as file:
                info = pickle.load(file)

            self.steps = info["steps"]
            self.best_reward_mean = info["best_reward_mean"]
            self.episode = info["episode"]

            logger.info("Continuing from %d episode (%d steps) with best reward mean %.2f" %
                        (self.episode, self.steps, self.best_reward_mean))

    def save(self, force=False):
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

        if (len(self.reward_history) >= self.network_config.save_steps and
                self.episode % self.network_config.save_steps == 0):

            total_reward = sum(self.reward_history[-self.network_config.save_steps:])
            current_reward_mean = total_reward / self.network_config.save_steps

            if current_reward_mean >= self.best_reward_mean:
                self.best_reward_mean = current_reward_mean
                info["best_reward_mean"] = current_reward_mean
                logger.info("Saving network. Found new best reward (%.2f)" % current_reward_mean)
                self.eval_model.save_network()
                self.target_model.save_network()
                with open(self.network_config.network_path + "/adaptive.info", "wb") as file:
                    pickle.dump(info, file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                logger.info("The best reward is still %.2f. Not saving" % current_reward_mean)

    def update(self):
        if len(self.memory) <= self.reinforce_config.batch_size:
            return

        beta = self.beta_schedule.value(self.steps)
        self.summary.add_scalar(tag='%s/Beta' % self.name,
                                scalar_value=beta, global_step=self.steps)

        batch = self.memory.sample(self.reinforce_config.batch_size, beta)

        (states, actions, reward, next_states,
         is_terminal, weights, batch_idxes) = batch

        self.summary.add_histogram(tag='%s/Batch Indices' % self.name,
                                   values=Tensor(batch_idxes),
                                   global_step=self.steps)

        states = Tensor(states)
        next_states = Tensor(next_states)
        terminal = FloatTensor(is_terminal)
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.reinforce_config.batch_size,
                                   dtype=torch.long)

        # Find the target values
        q_actions, q_values, _ = self.eval_model.predict_batch(states)
        q_values = q_values[:, batch_index, actions]
        _, q_next, _ = self.target_model.predict_batch(next_states)
        q_next = q_next.mean(2).detach()
        q_next = (1 - terminal) * q_next
        q_target = reward.t() + self.reinforce_config.discount_factor * q_next

        # Update the model
        fit_start_time = time.time()
        self.eval_model.fit(q_values, q_target, self.steps)
        self.fit_time += time.time() - fit_start_time

        # Update priorities
        td_errors = q_values - q_target
        td_errors = torch.sum(td_errors, 0)
        new_priorities = torch.abs(td_errors) + 1e-6  # prioritized_replay_eps
        self.memory.update_priorities(batch_idxes, new_priorities.data)
