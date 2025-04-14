import gymnasium as gym
from mc_network import PolicyNetwork, ValueFunction
from mc_algo import mc_without_baseline, mc_with_baseline
from utils import plot_reward_both, plot_reward
import wandb
import numpy as np

# Example usage for CartPole-v1
env = gym.make('CartPole-v1')
name = "CartPole"


# Initialize W&B
wandb.login()

# Define hyperparameters and their ranges
parameters_dict = {
    'algorithm': {'values': ['mc_reinforce_with_baseline']},
    'LR_policy': {'min': 3e-4, 'max': 9e-4},
    'LR_value': {'min': 1e-2, 'max': 9e-2},
    'GAMMA': {'value': 0.99},
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
sweep_id = wandb.sweep(sweep_config, project="mc_reinforce_with_baseline_CartPole_hyperparameter_finetuning_minimizing_regret")

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
        
        if config.algorithm == "mc_reinforce_with_baseline":
            # Rename the run based on hyperparameters
            run.name = f'{config.algorithm}_LR_policy{config.LR_policy}_LR_value_{config.LR_value}seed_{config.seed}'

            policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, learning_rate=config.LR_policy)
            value_network = ValueFunction(env.observation_space.shape[0], lr_value=config.LR_value)

            reward_list = mc_with_baseline(policy=policy,
                                        value_network=value_network,
                                        env=env,
                                        num_episodes=config.num_episodes,
                                        seed=config.seed,
                                        name=name,
                                        GAMMA=config.GAMMA)
                       
            for reward in reward_list:
                wandb.log({"Episode_return": reward})
                
        # wandb.finish()

# Run the sweep agent
wandb.agent(sweep_id, train)
