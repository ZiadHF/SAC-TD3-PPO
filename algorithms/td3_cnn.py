import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .td3 import TD3Agent
from utils.networks import NatureCNN, MLP, ConvTwinCritic
from utils.buffers import ReplayBuffer


class ConvDeterministicActor(nn.Module):
    """Deterministic actor for TD3 with CNN encoder"""
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dims=[256, 256]):
        super().__init__()
        self.cnn = NatureCNN(obs_shape, feature_dim)
        self.net = MLP(feature_dim, hidden_dims, action_dim, output_activation=nn.Tanh)
    
    def forward(self, obs):
        features = self.cnn(obs)
        return self.net(features)


class TD3AgentCNN(TD3Agent):
    def __init__(self, obs_shape, action_dim, feature_dim=512, **kwargs):
        self.device = kwargs['device']
        self.gamma = kwargs['gamma']
        self.tau = kwargs['tau']
        self.policy_noise = kwargs['policy_noise']
        self.noise_clip = kwargs['noise_clip']
        self.policy_delay = kwargs['policy_delay']
        self.batch_size = kwargs['batch_size']
        
        # Extract only the hidden_dims for networks
        hidden_dims = kwargs.get('hidden_dims', [256, 256])
        
        # CNN networks - TD3 uses DETERMINISTIC actor (not Gaussian!)
        self.actor = ConvDeterministicActor(obs_shape, action_dim, feature_dim, hidden_dims).to(self.device)
        self.actor_target = ConvDeterministicActor(obs_shape, action_dim, feature_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=kwargs['lr'])
        
        self.critic = ConvTwinCritic(obs_shape, action_dim, feature_dim, hidden_dims=hidden_dims).to(self.device)
        self.critic_target = ConvTwinCritic(obs_shape, action_dim, feature_dim, hidden_dims=hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=kwargs['lr'])
        
        self.replay_buffer = ReplayBuffer(kwargs['buffer_size'], obs_shape, action_dim)
        self.exploration_noise = kwargs.get('exploration_noise', 0.1)
        self.train_steps = 0

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        if not deterministic:
            action += np.random.normal(0, self.exploration_noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)