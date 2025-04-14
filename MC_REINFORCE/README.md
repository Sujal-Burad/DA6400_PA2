This is for MC_REINFORCE

The architecture details are mentioned in the report.


### Hyperparameter Configuration

The hyperparameters for the MC_REINFORCE_without_baseline algorithm are defined as follows:

| Hyperparameter | Type | Range/Values |
|----------------|------|--------------|
| **algorithm**   | Categorical | `mc_reinforce_without_baseline` |
| **LR_policy**       | Continuous | `[3e-4, 9e-4]` |
| **gamma**       | Fixed      | `0.99` |
| **num_episodes**| Fixed      | `1000` |
| **seed**        | Categorical | `[10, 20, 30, 40, 50]` |
| **render**      | Fixed      | `False` |

The hyperparameters for the MC_REINFORCE_with_baseline algorithm are defined as follows:

| Hyperparameter | Type | Range/Values |
|----------------|------|--------------|
| **algorithm**   | Categorical | `mc_reinforce_with_baseline` |
| **LR_policy**       | Continuous | `[3e-4, 9e-4]` |
| **LR_value**       | Continuous | `[1e-2, 9e-2]` |
| **gamma**       | Fixed      | `0.99` |
| **num_episodes**| Fixed      | `1000` |
| **seed**        | Categorical | `[10, 20, 30, 40, 50]` |
| **render**      | Fixed      | `False` |


The wandb hyperparameter_tuning plots for MC_reinforce without_baseline are:
1.  [Cartpole-v1](https://wandb.ai/sujal/mc_reinforce_without_baseline_CartPole_hyperparameter_finetuning_minimizing_regret/sweeps/fyh7gmsa)
2.  [Acrobot-v1](https://wandb.ai/sujal/mc_reinforce_without_baseline_Acrobot_hyperparameter_finetuning_minimizing_regret/sweeps/76iqjfah)

The wandb hyperparameter_tuning plots for MC_reinforce with_baseline are:
1.  [Cartpole-v1](https://wandb.ai/sujal/mc_reinforce_with_baseline_CartPole_hyperparameter_finetuning_minimizing_regret/sweeps/20uceylp)
2.  [Acrobot-v1](https://wandb.ai/sujal/mc_reinforce_with_baseline_Acrobot_hyperparameter_finetuning_minimizing_regret/sweeps/pt4kifww)
