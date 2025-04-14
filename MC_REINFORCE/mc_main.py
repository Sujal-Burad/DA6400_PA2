import gymnasium as gym
from mc_network import PolicyNetwork, ValueFunction
from mc_algo import mc_without_baseline, mc_with_baseline
from utils import plot_reward_both, plot_reward

# Example usage for Acrobot-v1
env = gym.make('Acrobot-v1')
name = "Acrobot"

GAMMA = 0.99            # discount factor
LR_policy = 3e-4               # learning rate for policy network
LR_value = 1e-2               # learning rate for value network
num_episodes = 1000    # Number of episodes


reward_list_seed_without_baseline = []
reward_list_seed_with_baseline = []

# mcc without baseline
for seed in [10, 20, 30, 40, 50]:
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, learning_rate=LR_policy)
    reward_list = mc_without_baseline(policy=policy,
                                      env=env,
                                      num_episodes=num_episodes,
                                      seed=seed,
                                      name=name,
                                      GAMMA=GAMMA)
    
    reward_list_seed_without_baseline.append(reward_list)

# mc with baseline
for seed in [10, 20, 30, 40, 50]:
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, learning_rate=LR_policy)
    value_network = ValueFunction(env.observation_space.shape[0], lr_value=LR_value)

    reward_list = mc_with_baseline(policy=policy,
                                   value_network=value_network,
                                    env=env,
                                    num_episodes=num_episodes,
                                    seed=seed,
                                    name=name,
                                    GAMMA=GAMMA)
    reward_list_seed_with_baseline.append(reward_list)

plot_reward_both(reward_list_seed_avg=reward_list_seed_without_baseline,
            reward_list_seed_max=reward_list_seed_with_baseline,
            num_episodes=num_episodes,
            file_name=f'mc_{name}.png')

# print("reward_list_seed_without_baseline = ", reward_list_seed_without_baseline)
# print("num_episodes = ", num_episodes)
plot_reward(reward_list_seed=reward_list_seed_without_baseline,
            num_episodes=num_episodes,
            file_name=f'mc_without_baseline_{name}.png')

plot_reward(reward_list_seed=reward_list_seed_with_baseline,
            num_episodes=num_episodes,
            file_name=f'mc_with_baseline_{name}.png')
