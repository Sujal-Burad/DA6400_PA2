import os 
import matplotlib.pyplot as plt
import numpy as np
from dueling_dqn_agent import Agent
import gymnasium as gym

def plot_reward_both(reward_list_seed_avg, reward_list_seed_max, num_episodes, path="./", file_name="average_reward.png"):
    """
        Description       : This function plots the average reward.
        Args:
            reward_list_seed    : The list of rewards for 5 seeds.
            num_episodes        : The number of episodes.
            path                : The path for saving the plot.
            filename            : The name of the plot.
    """
    # Calculate mean and variance across seeds for each episode
    mean_rewards_avg = np.mean(reward_list_seed_avg, axis=0)
    variance_rewards_avg = np.var(reward_list_seed_avg, axis=0)
    plt.plot(np.arange(0, num_episodes), mean_rewards_avg, label='Mean Reward (without_baseline)', color='blue')
    
    # Plot variance as a shaded region around the mean
    plt.fill_between(np.arange(0, num_episodes), mean_rewards_avg - np.sqrt(variance_rewards_avg), 
                     mean_rewards_avg + np.sqrt(variance_rewards_avg), alpha=0.2, label='Variance (without_baseline)', color='blue')

    mean_rewards_max = np.mean(reward_list_seed_max, axis=0)
    variance_rewards_max = np.var(reward_list_seed_max, axis=0)
    plt.plot(np.arange(0, num_episodes), mean_rewards_max, label='Mean Reward (with_baseline)', color='red')
    
    # Plot variance as a shaded region around the mean
    plt.fill_between(np.arange(0, num_episodes), mean_rewards_max - np.sqrt(variance_rewards_max), 
                     mean_rewards_max + np.sqrt(variance_rewards_max), alpha=0.2, label='Variance (with_baseline)', color='red')

    plt.xlabel("Episodes")
    plt.ylabel("Episodic return")
    plt.legend()
    plt.savefig(os.path.join(path, file_name))
    plt.close()

def plot_reward(reward_list_seed, num_episodes, path="./", file_name="average_reward.png"):
    """
        Description       : This function plots the average reward.
        Args:
            reward_list_seed    : The list of rewards for 5 seeds.
            num_episodes        : The number of episodes.
            path                : The path for saving the plot.
            filename            : The name of the plot.
    """
    # Calculate mean and variance across seeds for each episode
    mean_rewards_avg = np.mean(reward_list_seed, axis=0)
    variance_rewards_avg = np.var(reward_list_seed, axis=0)
    # print("mean_reward_avg = ", mean_rewards_avg)
    # print("variance_rewards_avg = ", variance_rewards_avg)
    plt.plot(np.arange(0, num_episodes), mean_rewards_avg, label='Mean Reward', color='blue')
    
    # Plot variance as a shaded region around the mean
    plt.fill_between(np.arange(0, num_episodes), mean_rewards_avg - np.sqrt(variance_rewards_avg), 
                     mean_rewards_avg + np.sqrt(variance_rewards_avg), alpha=0.2, label='Variance', color='blue')


    plt.xlabel("Episodes")
    plt.ylabel("Episodic return")
    plt.legend()
    plt.savefig(os.path.join(path, file_name))
    plt.close()

def render_policy(name, env, agent):
    agent.load_model(f'dueling_dqn_{name}_{agent.adv_type}')
    state,_ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        env.render()
        if done:
            break
        state = next_state

    print("Total_reward in render = ", total_reward)
    env.close()
