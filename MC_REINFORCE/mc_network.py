import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Define the policy network
# class Policy(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Policy, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, action_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.softmax(self.fc3(x), dim=1)  # Use dim=1 for softmax
#         return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = action_dim
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear2(F.relu(self.linear1(state))))
        x = F.softmax(self.linear3(x), dim=1)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
    

# Define the value function network (baseline)
class ValueFunction(nn.Module):
    def __init__(self, state_dim, lr_value):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_value)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    