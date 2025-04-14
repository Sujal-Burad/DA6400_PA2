import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from q_network import QNetwork
from replay_buffer import ReplayBuffer

class Agent():

    def __init__(self, state_size, action_size, seed, adv_type, device, BUFFER_SIZE, BATCH_SIZE, LR, GAMMA, UPDATE_EVERY):

        ''' Agent Environment Interaction '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.adv_type = adv_type
        self.device = device
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY


        ''' Q-Network '''
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed, self.adv_type).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed, self.adv_type).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        ''' Replay memory '''
        self.memory = ReplayBuffer(self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.seed, self.device)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)

        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA)

        """ +Q TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        ''' Epsilon-greedy action selection (Already Present) '''
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # def act_softmax(self, state, tau=1.0):

    #     state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    #     self.qnetwork_local.eval()
    #     with torch.no_grad():
    #         action_values = self.qnetwork_local(state)
    #     self.qnetwork_local.train()

    #     ''' Softmax action selection '''
    #     x = [action_value/tau for action_value in action_values.cpu().data.numpy()][0]
    #     return np.random.choice(np.arange(self.action_size), p=softmax(x))

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted Q values (for next states) from target model'''
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        ''' Compute Q targets for current states '''
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        ''' Get expected Q values from local model '''
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ''' Compute loss '''
        loss = F.mse_loss(Q_expected, Q_targets)

        ''' Minimize the loss '''
        self.optimizer.zero_grad()
        loss.backward()

        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            if param.grad != None:
              param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        
    def save_model(self,path):
        torch.save(self.qnetwork_local.state_dict(), f'weights/{path}.pth')
        
    def load_model(self,path):
        self.qnetwork_local.load_state_dict(torch.load(f'weights/{path}.pth', map_location=torch.device(self.device)))
        self.qnetwork_local.eval()