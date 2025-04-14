For Dueling-DQN


The architecture details are mentioned in the report.


### Hyperparameter Configuration

The hyperparameters for the Dueling_DQN_type1 algorithm are defined as follows:

| Hyperparameter | Type | Range/Values |
|----------------|------|--------------|
| **algorithm**   | Categorical | `dualing_dqn_type1` |
| **BUFFER_SIZE**       | Continuous | `[int(1e2), int(1e7)]` |
| **BATCH_SIZE**       | Categorical | `[8, 16, 32, 64, 128, 256, 512][3e-4, 9e-4]` |
| **LR**       | Continuous | `[5e-4, 1.0]` |
| **epsilon_start**       | Continuous | `[0.2, 1.0]` |
| **epsilon_end**       | Continuous | `[0.001, 0.15]` |
| **epsilon_decay**       | Continuous | `[0.9, 0.99]` |
| **gamma**       | Fixed      | `0.99` |
| **UPDATE_EVERY**        | Categorical | `[10, 20, 30, 50, 80, 100]` |
| **num_episodes**| Fixed      | `1000` |
| **seed**        | Categorical | `[10, 20, 30, 40, 50]` |
| **render**      | Fixed      | `False` |



The hyperparameters for the Dueling_DQN_type2 algorithm are defined as follows:

| Hyperparameter | Type | Range/Values |
|----------------|------|--------------|
| **algorithm**   | Categorical | `dualing_dqn_type2` |
| **BUFFER_SIZE**       | Continuous | `[int(1e2), int(1e7)]` |
| **BATCH_SIZE**       | Categorical | `[8, 16, 32, 64, 128, 256, 512][3e-4, 9e-4]` |
| **LR**       | Continuous | `[5e-4, 1.0]` |
| **epsilon_start**       | Continuous | `[0.2, 1.0]` |
| **epsilon_end**       | Continuous | `[0.001, 0.15]` |
| **epsilon_decay**       | Continuous | `[0.9, 0.99]` |
| **gamma**       | Fixed      | `0.99` |
| **UPDATE_EVERY**        | Categorical | `[10, 20, 30, 50, 80, 100]` |
| **num_episodes**| Fixed      | `1000` |
| **seed**        | Categorical | `[10, 20, 30, 40, 50]` |
| **render**      | Fixed      | `False` |


The wandb hyperparameter_tuning plots for Dueling_DQN type1 are:
1.  [Cartpole-v1](https://wandb.ai/sujal/dualing_dqn_type1_cartpole_hyperparameter_finetuning_minimizing_regret/sweeps/aet65glq)
2.  [Acrobot-v1](https://wandb.ai/sujal/dualing_dqn_type1_Acrobot_hyperparameter_finetuning_minimizing_regret/sweeps/5t7wyxi8)

The wandb hyperparameter_tuning plots for Dueling_DQN type2 are:
1.  [Cartpole-v1](https://wandb.ai/sujal/dualing_dqn_type2_cartpole_hyperparameter_finetuning_minimizing_regret/sweeps/lhu49mex)
2.  [Acrobot-v1](https://wandb.ai/sujal/dualing_dqn_type2_Acrobot_hyperparameter_finetuning_minimizing_regret/sweeps/cntwmiz7)
