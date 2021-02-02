import numpy as np
import random
from collections import namedtuple, deque

from model import DuelingDQN
from replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
TAU = 1e-3              # soft update


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 gamma=GAMMA,
                 buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 update_every=UPDATE_EVERY,
                 lr=LR,
                 tau=TAU
    ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.batch_size = batch_size

        # Q-Network
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_local = DuelingDQN(state_size, action_size, seed).to(self.device)
        self.model_target = DuelingDQN(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.model_local.parameters(), lr=LR)
    
        # Replay memory
        self.memory = ReplayBuffer(
            action_size=action_size,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            seed=seed,
            device=self.device
        )
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.update(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        
        self.model_local.eval()
        with torch.no_grad():
            qvals = self.model_local.forward(state)
        self.model_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(qvals.cpu().detach().numpy())
            return action
        else:
            return random.choice(np.arange(self.action_size))
    

    def update(self, batch):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            batch (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = batch
        
        # Get expected Q values from local model
        curr_Q = self.model_local.forward(states).gather(1, actions)
#         curr_Q = curr_Q.squeeze(1)
        
        # Get max predicted Q values (for next states) from target model
        max_next_Q = self.model_target.forward(next_states).detach().max(1)[0].unsqueeze(1)
        expected_Q = rewards + (self.gamma * max_next_Q * (1 - dones))

        loss = F.mse_loss(curr_Q, expected_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target model
        self.update_target(self.model_local, self.model_target, TAU)     
    def update_target(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)