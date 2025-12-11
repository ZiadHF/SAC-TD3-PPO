import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base import BaseAgent
from utils.networks import GaussianPolicy, TwinCritic
from utils.buffers import ReplayBuffer

class SACAgent(BaseAgent):
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.2, buffer_size=500000, batch_size=256, 
                 hidden_dims=[256, 256], automatic_entropy_tuning=True, 
                 target_entropy=None, device='cuda'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        self.actor = GaussianPolicy(obs_dim, action_dim, hidden_dims).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = TwinCritic(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target = TwinCritic(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        
        self.replay_buffer = ReplayBuffer(buffer_size, (obs_dim,), action_dim)

    def select_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.actor.sample(obs, deterministic)
        return action.cpu().numpy()[0]

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return None
            
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(batch['next_obs'])
            min_q_next = self.critic_target.min_q(batch['next_obs'], next_action)
            q_target = batch['rewards'].unsqueeze(1) + (1 - batch['dones'].unsqueeze(1)) * self.gamma * (min_q_next - self.alpha * next_log_prob)
        
        q1, q2 = self.critic(batch['obs'], batch['actions'])
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optim.step()
        
        action_pred, log_prob, _ = self.actor.sample(batch['obs'])
        q_new = self.critic.min_q(batch['obs'], action_pred)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optim.step()
        
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'entropy': -log_prob.mean().item()
        }