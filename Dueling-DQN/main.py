import torch
import datetime
import gymnasium as gym


from dueling_dqn_agent import Agent
from utils import plot_reward_both, render_policy, plot_reward
from dqn_algo import dqn

env = gym.make('Acrobot-v1')
name = 'Acrobot'
print("env = ", env)
print("name = ", name)
# Solving with epsilon-greedy strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 9e-1               # learning rate
UPDATE_EVERY = 20       # how often to update the network (When Q target is present)
num_episodes = 100    # Number of episodes
epsilon_start = 1.0     # Starting epsilon
epsilon_end = 0.01      # Ending (least) value of epsilon 
epsilon_decay = 0.905   # Epsilon decay

# maximum number of steps taken in the cartpole and acrobot environment is determined by the terminated, and truncated variables in the document
# Epsilon decay is after every 100 episodes

begin_time = datetime.datetime.now()
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n



# Type 1
print("Type 1 of Dueling-DQN, average")
reward_list_seed_avg = []
for seed in [10, 20, 30, 40, 50]:
    print("seed = ", seed)
    agent = Agent(state_size=state_shape, action_size=action_shape, seed=seed, device=device, adv_type="avg", BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, LR=LR, GAMMA=GAMMA, UPDATE_EVERY=UPDATE_EVERY)
    rewards = dqn(name=name,
                env = env,
                agent=agent,
                n_episodes=num_episodes, 
                eps_start=epsilon_start,
                eps_end=epsilon_end,
                eps_decay=epsilon_decay,
                seed=seed)
    reward_list_seed_avg.append(rewards)

    # render_policy(name=name, env=env, agent=agent)

    
time_taken = datetime.datetime.now() - begin_time
print('Time taken for average = ', time_taken)

# Type 2
begin_time = datetime.datetime.now()
print("Type 2 of Dueling-DQN, maximum")
reward_list_seed_max = []
for seed in [10, 20, 30, 40, 50]:
    print("seed = ", seed)
    agent = Agent(state_size=state_shape, action_size=action_shape, seed=seed, device=device, adv_type="max", BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, LR=LR, GAMMA=GAMMA, UPDATE_EVERY=UPDATE_EVERY)
    rewards = dqn(name=name,
                env = env,
                agent=agent,
                n_episodes=num_episodes, 
                eps_start=epsilon_start,
                eps_end=epsilon_end,
                eps_decay=epsilon_decay,
                seed=seed)
    reward_list_seed_max.append(rewards)
    
    # render_policy(name=name, env=env, agent=agent)

time_taken = datetime.datetime.now() - begin_time
print('Time taken for maximum = ', time_taken)

# print("average reward list = ", reward_list_seed_avg)
# print("maximum reward list = ", reward_list_seed_max)
plot_reward_both(reward_list_seed_avg=reward_list_seed_avg,
            reward_list_seed_max=reward_list_seed_max,
            num_episodes=num_episodes,
            file_name=f'dueling_dqn_{name}.png')

plot_reward(reward_list_seed=reward_list_seed_max,
            num_episodes=num_episodes,
            file_name=f'dueling_dqn_{name}_max.png')

plot_reward(reward_list_seed=reward_list_seed_avg,
            num_episodes=num_episodes,
            file_name=f'dueling_dqn_{name}_avg.png')
