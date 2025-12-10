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
        # Ensure channel dimension is passed correctly to NatureCNN
        # obs_shape is likely (84, 84, 4) or (4, 84, 84). We handle the channel count.
        c = obs_shape[0] if obs_shape[0] < obs_shape[2] else obs_shape[2]
        self.cnn = NatureCNN(c, feature_dim) 
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
        
        hidden_dims = kwargs.get('hidden_dims', [256, 256])
        
        # CNN networks
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
        # 1. Convert to float tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 2. CRITICAL: Normalize Image [0, 255] -> [0, 1]
        obs_tensor = obs_tensor / 255.0
        

        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
            
        if not deterministic:
            # Add noise for exploration
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            
        return np.clip(action, -1.0, 1.0)

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return None
        
        self.train_steps += 1
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        
        # --- PREPROCESSING START ---
        # 1. Normalize
        obs = batch['obs'].float() / 255.0
        next_obs = batch['next_obs'].float() / 255.0
            
        # 3. Scale Rewards (Stabilizes Critic)
        rewards = batch['rewards'] / 20.0
        # --- PREPROCESSING END ---

        # Critic Update
        with torch.no_grad():
            # Select action according to target actor and add clipped noise
            noise = (torch.randn_like(batch['actions']) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + (1 - batch['dones'].unsqueeze(1)) * self.gamma * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(obs, batch['actions'])

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        actor_loss = None
        
        # Delayed policy updates
        if self.train_steps % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1(obs, self.actor(obs)).mean()
            
            # Optimize the actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Return metrics (check if actor_loss was computed)
        metrics = {'critic_loss': critic_loss.item()}
        if actor_loss is not None:
            metrics['actor_loss'] = actor_loss.item()
            
        return metrics