import numpy as np
import torch

def update_policy(GAMMA, policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


def mc_without_baseline(policy, env, num_episodes, seed, name, GAMMA):
    reward_list = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed)
        # print("state = ", state, type(state))
        rewards = []
        log_probs = []

        done = False
        score = 0
        while not done:
            action, log_prob = policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

            if done:
                break
        update_policy(GAMMA=GAMMA, policy_network=policy, rewards=rewards, log_probs=log_probs)

        reward_list.append(score)

        # print("Episode = ", episode, end="")

        print('Episode {}\t Score: {:.2f}'.format(episode, score))

        # if i_episode % 100 == 0:
        #    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if score>=np.max(reward_list):
           print(f'New Best score is for episode {episode}, and is equal to {score}')
           print("Average score so far = ", np.mean(reward_list))
           print(f'Model saved as mc_reinforce_without_baseline_{name}')
           torch.save(policy.state_dict(), f'weights/mc_reinforce_without_baseline_{name}.pth')

    return reward_list








def update_policy_with_baseline(GAMMA, policy_network, value_network, rewards, log_probs, states):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()

    values = value_network(torch.from_numpy(np.array(states)).to(torch.float32))
    td_errors = []
    for i in range(len(states) - 1):
        td_errors.append(rewards[i] + GAMMA * values[i + 1] - values[i])
    
    td_errors.append(rewards[len(states) - 1] - values[len(states) - 1])
    # td_errors = torch.from_numpy(np.array(td_errors)).to(torch.float32)
    # td_errors = torch.tensor(td_errors)

    # print("td_erros = ", td_errors, type(td_errors), td_errors.shape)

    value_network.optimizer.zero_grad()
    value_gradient = torch.stack(td_errors).sum()
    value_gradient.backward()
    value_network.optimizer.step()



def mc_with_baseline(policy, value_network, env, num_episodes, seed, name, GAMMA):
    reward_list = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed)

        states = []
        rewards = []
        log_probs = []

        done = False
        score = 0
        while not done:
            states.append(state)
            action, log_prob = policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

            if done:
                break

        update_policy_with_baseline(GAMMA=GAMMA, policy_network=policy, value_network=value_network, rewards=rewards, log_probs=log_probs, states=states)

        reward_list.append(score)

        # print("Episode = ", episode, end="")

        print('Episode {}\t Score: {:.2f}'.format(episode, score))

        # if i_episode % 100 == 0:
        #    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if score>=np.max(reward_list):
           print(f'New Best score is for episode {episode}, and is equal to {score}')
           print("Average score so far = ", np.mean(reward_list))
           print(f'Model saved as mc_reinforce_with_baseline_{name}')
           torch.save(policy.state_dict(), f'weights/mc_reinforce_with_baseline_{name}.pth')

    return reward_list