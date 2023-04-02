import numpy as np
import gym
from stable_baselines3.common import utils

logger = utils.configure_logger(tensorboard_log='./tensor-logs-2', tb_log_name = 'run')
env = gym.make("FrozenLake-v1")
n_observations = env.observation_space.n
n_actions = env.action_space.n

Q_table = np.zeros((n_observations, n_actions))

n_episodes = 10000
max_iter_episode = 100
exploration_proba = 1
exploration_decreasing_decay = 0
min_exploration_proba = 0.01
gamma = 0.99
lr = 0.1

total_rewards_episode = []
n_timesteps = 0

#we iterate over episodes
for e in range(n_episodes):
    current_state = env.reset()
    done = False
    total_episode_reward = 0

    for i in range(max_iter_episode):
        n_timesteps += 1

        if np.random.uniform(0,1) < 0.2:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[current_state,:])

        next_state, reward, done, _ = env.step(action)

        Q_table[current_state, action] = (1-lr) * Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:]))
        total_episode_reward += reward

        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state

    exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
    total_rewards_episode.append(total_episode_reward)

    if e % 100 == 0 and e > 0:
        mean_reward = np.mean(total_rewards_episode[-100:])

        print(n_timesteps, mean_reward)

        logger.record('reward', mean_reward)
        logger.dump(step = e)