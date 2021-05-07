# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
import logging
import datetime
import gym
import numpy as np
import tensorflow as tf
import itertools
from tensorflow import keras
from tensorflow.keras import layers
import multiprocessing

from lords_ai.q_model import create_q_model
from lord_gym.envs.lord_env import LordEnv  # does registration
logging.basicConfig(level=logging.ERROR)


optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

class DeepAgent():
    seed = 42
    gamma = 0.1  # Discount factor for past rewards max 0.99
    delta = 0.1  # Discount factor for past rewards mean 0.99
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
    batch_size = 32  # Size of batch taken from replay buffer
    max_steps_per_episode = 10000

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    no_action_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50  # 50000
    # Number of frames for exploration
    epsilon_greedy_frames = 1000.0  # 1000000
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 100
    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    def __init__(self, gamma, delta):
        self.gamma = gamma
        self.gamma = delta
        self.env = gym.make('lord-v0')

        self.num_actions = self.env.action_space.n
        self.max_steps_per_episode = self.env.max_rounds

        # The first model makes the predictions for Q-values which are used to
        # make a action.
        self.model = create_q_model(self.num_actions)
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        self.model_target = create_q_model(self.num_actions)

    def run(self):
        while True:  # Run until solved
            state = np.array(self.env.reset())
            episode_reward = 0

            for timestep in range(1, self.max_steps_per_episode + 1):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.
                self.frame_count += 1

                # Use epsilon-greedy for exploration
                if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                    # Take random action
                    action = np.random.choice(self.num_actions)
                else:
                    # Predict action Q-values
                    # From environment state
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = self.model(state_tensor, training=False)
                    # Take best action
                    action = tf.argmax(action_probs[0]).numpy()

                # Decay probability of taking random action
                self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
                self.epsilon = max(self.epsilon, self.epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done, _ = self.env.step(action)
                state_next = np.array(state_next)  # TODO process resources, add flag for each "buyable" elements, add weekly income variable

                episode_reward += reward

                # Save actions and states in replay buffer
                self.action_history.append(action)
                self.state_history.append(state)
                self.state_next_history.append(state_next)
                self.done_history.append(done)
                self.rewards_history.append(reward)
                self.no_action_history.append(all(state == state_next))
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if self.frame_count % self.update_after_actions == 0 and len(self.done_history) > self.batch_size:

                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([self.state_history[i] for i in indices])
                    state_next_sample = np.array([self.state_next_history[i] for i in indices])
                    rewards_sample = [self.rewards_history[i] for i in indices]
                    action_sample = [self.action_history[i] for i in indices]
                    no_action_sample = tf.convert_to_tensor([float(self.no_action_history[i]) for i in indices])
                    done_sample = tf.convert_to_tensor([float(self.done_history[i]) for i in indices])

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = self.model_target.predict(state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1) + self.delta * tf.reduce_mean(future_rewards, axis=1)

                    # If final frame set the last value to -1
                    # TODO think about this, because this says done is bad, but actually done is only bad if lost
                    # updated_q_values = updated_q_values * (1 - done_sample) - done_sample
                    updated_q_values = updated_q_values * (1 - no_action_sample) - no_action_sample  # replace value for no action with -1

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, self.num_actions)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        #x = np.asarray(x).astype('float32')
                        q_values = self.model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = self.loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if self.frame_count % self.update_target_network == 0:
                    # update the the target network with new weights
                    self.model_target.set_weights(self.model.get_weights())
                    # Log details
                    print(f"running reward: {running_reward:.2f} at episode {self.episode_count}, frame count {self.frame_count}, no action {running_no_action} {self.episode_reward_history}")

                # Limit the state and reward history
                if len(self.rewards_history) > self.max_memory_length:
                    del self.rewards_history[:1]
                    del self.state_history[:1]
                    del self.state_next_history[:1]
                    del self.action_history[:1]
                    del self.done_history[:1]

                if done:
                    break

            # Update running reward to check condition for solving
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]
            running_reward = np.mean(self.episode_reward_history)
            running_no_action = round(np.sum(self.no_action_history) / len(self.no_action_history), 2)

            self.episode_count += 1

            if self.episode_count > 3 and running_reward > 40:  # Condition to consider the task solved
                print("Solved at episode {}!".format(self.episode_count))
                break
            if self.episode_count > 5000:
                break

        return self.episode_count, running_reward, self.episode_reward_history
        #version = datetime.datetime.now()
        #print(f"Model saving - {version}")
        #self.model.save(f'../models/{version}_model.h5')
        #self.model_target.save(f'../models/{version}_model_target.h5')


def run(params):
    da = DeepAgent(params[0], params[1])
    episode_count, running_reward, episode_reward_history = da.run()
    return params[0], params[1], episode_count, running_reward, episode_reward_history


if __name__ == '__main__':
    # with multiprocessing.Pool(2) as pool:
    #     results = pool.map(run, itertools.permutations([0.0, 0.4, 0.6, 0.9], r=2))
    #     for r in results:
    #         print(r)
    episode_count, running_reward, episode_reward_history = DeepAgent(gamma=0.2, delta=0.8).run()