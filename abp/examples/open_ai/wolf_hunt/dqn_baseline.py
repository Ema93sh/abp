import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import BatchInput
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=100, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=100, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    with U.make_session(8):
        # Create the environment
        env = gym.make("WolfHunt-v0")
        # Create all the functions necessary to train the model
        wolf1_act, wolf1_train, wolf1_update_target, wolf1_debug = deepq.build_train(
            make_obs_ph=lambda name: BatchInput(147, name=name),
            q_func=model,
            num_actions=5,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        wolf2_act, wolf2_train, wolf2_update_target, wolf2_debug = deepq.build_train(
            make_obs_ph=lambda name: BatchInput(147, name=name),
            q_func=model,
            num_actions=5,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        # Create the replay buffer
        wolf1_replay_buffer = ReplayBuffer(50000)
        wolf2_replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        wolf1_update_target()
        wolf2_update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            wolf1_action = wolf2_act(obs[None], update_eps=exploration.value(t))[0]
            wolf2_action = wolf2_act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step((wolf1_action, wolf2_action))

            # Store transition in the replay buffer.
            wolf1_replay_buffer.add(obs, wolf1_action, rew, new_obs, float(done))
            wolf2_replay_buffer.add(obs, wolf2_action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = wolf1_replay_buffer.sample(32)
                    wolf1_train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                    obses_t, actions, rewards, obses_tp1, dones = wolf2_replay_buffer.sample(32)
                    wolf2_train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    wolf1_update_target()
                    wolf2_update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
