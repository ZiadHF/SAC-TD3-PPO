import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base import BaseAgent
from utils.networks import MLP, TwinCritic
from utils.buffers import ReplayBuffer
import numpy as np

class TD3Agent(BaseAgent):
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 buffer_size=500000, batch_size=256, hidden_dims=[256, 256], 
                 exploration_noise=0.1, device='cuda'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        
        self.actor = MLP(obs_dim, hidden_dims, action_dim, output_activation=nn.Tanh).to(device)
        self.actor_target = MLP(obs_dim, hidden_dims, action_dim, output_activation=nn.Tanh).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = TwinCritic(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target = TwinCritic(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size, (obs_dim,), action_dim)
        self.exploration_noise = exploration_noise
        self.train_steps = 0

    def select_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]
        if not deterministic:
            action += np.random.normal(0, self.exploration_noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return None
            
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        
        with torch.no_grad():
            noise = torch.randn_like(batch['actions']) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(batch['next_obs']) + noise
            next_action = torch.clamp(next_action, -1.0, 1.0)
            
            q1_target, q2_target = self.critic_target(batch['next_obs'], next_action)
            min_q_target = torch.min(q1_target, q2_target)
            q_target = batch['rewards'].unsqueeze(1) + (1 - batch['dones'].unsqueeze(1)) * self.gamma * min_q_target
        
        q1, q2 = self.critic(batch['obs'], batch['actions'])
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optim.step()
        
        actor_loss = torch.tensor(0.0, device=self.device)
        if self.train_steps % self.policy_delay == 0:
            actor_loss = -self.critic.q1(batch['obs'], self.actor(batch['obs'])).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optim.step()
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.train_steps += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }