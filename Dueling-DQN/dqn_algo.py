import numpy as np

''' Defining DQN Algorithm '''


def dqn(name, env, agent, n_episodes=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.905, seed=10):
    rewards = []

    eps = eps_start
    ''' initialize epsilon '''

    for i_episode in range(1, n_episodes+1):
        state,_ = env.reset(seed=seed)
        score = 0
        done = False
        while not done:
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        rewards.append(score)
        if i_episode % 100 == 0:
            eps = max(eps_end, eps_decay*eps)
        ''' decrease epsilon '''

        print('Episode {}\t Score: {:.2f}'.format(i_episode, score))

        # if i_episode % 100 == 0:
        #    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if score>=np.max(rewards):
           print(f'New Best score is for episode {i_episode}, and is equal to {score}')
           print("Average score so far = ", np.mean(rewards))
           print(f'Model saved as dueling_dqn_{name}_{agent.adv_type}')
           agent.save_model(f'dueling_dqn_{name}_{agent.adv_type}')
        #    break
    return rewards