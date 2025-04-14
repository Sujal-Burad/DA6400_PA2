import torch
import datetime
import gymnasium as gym
import wandb
import numpy as np

from dueling_dqn_agent import Agent
from dqn_algo import dqn

name = 'CartPole_type2_hyperparameter_tuning'
print("name = ", name)

# Initialize W&B
wandb.login()

# Define hyperparameters and their ranges
parameters_dict = {
    'algorithm': {'values': ['dualing_dqn_type2']},
    'BUFFER_SIZE': {'min': int(1e2), 'max': int(1e7)},
    'BATCH_SIZE': {'values': [8, 16, 32, 64, 128, 256, 512]},
    'LR': {'min': 5e-4, 'max': 1.0},
    'epsilon_start': {'min': 0.2, 'max': 1.0},
    'epsilon_end': {'min': 0.001, 'max': 0.15},
    'epsilon_decay': {'min': 0.9, 'max': 0.99},
    'GAMMA': {'value': 0.99},
    'UPDATE_EVERY': {'values': [10, 20, 30, 50, 80, 100]},
    'num_episodes': {'value': 1000},
    'seed': {'values': [10, 20, 30, 40, 50]},
    'render': {'value': False}
}

# Create sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'Episode_return',
        'goal': 'maximize'
    },
    'parameters': parameters_dict
}


# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="dualing_dqn_type2_cartpole_hyperparameter_finetuning_minimizing_regret")

# Define training function
def train(config=None):
    # Initialize a new W&B run
    with wandb.init(config=config) as run:
        config = wandb.config
        
        # Set seed
        np.random.seed(config.seed)
        
        # define the environment
        if config.render: 
            env = gym.make('CartPole-v1', render_mode='human')
        else:
            env = gym.make('CartPole-v1')
        
        if config.algorithm == "dualing_dqn_type2":
            state_shape = env.observation_space.shape[0]
            action_shape = env.action_space.n
            # Rename the run based on hyperparameters
            run.name = f'{config.algorithm}_BUFFER_SIZE_{config.BUFFER_SIZE}_BATCH_SIZE_{config.BATCH_SIZE}_LR_{config.LR}_epsilon_start_{config.epsilon_start}_epsilon_end_{config.epsilon_end}_epsilon_decay_{config.epsilon_decay}_UPDATE_EVERY_{config.UPDATE_EVERY}_seed_{config.seed}'

            agent = Agent(state_size=state_shape, action_size=action_shape, seed=config.seed, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), adv_type="max", BUFFER_SIZE=int(config.BUFFER_SIZE), BATCH_SIZE=config.BATCH_SIZE, LR=config.LR, GAMMA=config.GAMMA, UPDATE_EVERY=config.UPDATE_EVERY)
            reward_list = dqn(name=name,
                        env = env,
                        agent=agent,
                        n_episodes=config.num_episodes, 
                        eps_start=config.epsilon_start,
                        eps_end=config.epsilon_end,
                        eps_decay=config.epsilon_decay,
                        seed=config.seed)

            
            for reward in reward_list:
                wandb.log({"Episode_return": reward})
                
        # wandb.finish()

# Run the sweep agent
wandb.agent(sweep_id, train)
