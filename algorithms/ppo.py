import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base import BaseAgent
from utils.networks import GaussianPolicy, MLP
from utils.buffers import PPORolloutBuffer
import numpy as np

class PPOAgent(BaseAgent):
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, clip_ratio=0.2, 
                 lam=0.95, train_pi_iters=80, train_v_iters=80, target_kl=0.01,
                 hidden_dims=[64, 64], max_ep_len=1000, ent_coef=0.0, batch_size=64, device='cuda'):
        self.device = device
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.lam = lam
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.max_ep_len = max_ep_len
        self.ent_coef = ent_coef
        self.batch_size = batch_size
        
        self.actor = GaussianPolicy(obs_dim, action_dim, hidden_dims).to(device)
        self.critic = MLP(obs_dim, hidden_dims, 1).to(device)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.buffer = PPORolloutBuffer()

    def select_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(obs, deterministic)
            value = self.critic(obs)
        # log_prob is None when deterministic=True
        if log_prob is None:
            return action.cpu().numpy()[0], 0.0, value.cpu().numpy()[0].item()
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0].item(), value.cpu().numpy()[0].item()

    def compute_gae(self, rewards, values, dones):
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages

    def train_step(self):
        # Train when called - train.py handles the timing based on rollout_length
        if len(self.buffer.obs) == 0:
            return None
            
        data = self.buffer.get()
        returns, advantages = self.compute_gae(data['rewards'], data['values'], data['dones'])
        
        old_obs = torch.as_tensor(data['obs'], dtype=torch.float32, device=self.device)
        old_actions = torch.as_tensor(data['actions'], dtype=torch.float32, device=self.device)
        old_logps = torch.as_tensor(data['logps'], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_losses = []
        critic_losses = []
        kl_divs = []
        entropies = []
        
        dataset_size = len(data['obs'])
        indices = np.arange(dataset_size)
        
        for i in range(self.train_pi_iters):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                batch_obs = old_obs[idx]
                batch_act = old_actions[idx]
                batch_logp = old_logps[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]
                
                new_logps, entropy = self.actor.evaluate_actions(batch_obs, batch_act)
                ratio = torch.exp(new_logps - batch_logp.unsqueeze(1))
                surr1 = ratio * batch_adv.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_adv.unsqueeze(1)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                actor_loss = -torch.min(surr1, surr2).mean() + self.ent_coef * entropy_loss
                
                values = self.critic(batch_obs).squeeze()
                critic_loss = F.mse_loss(values, batch_ret)
                
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optim.step()
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optim.step()
                
                kl = (batch_logp.unsqueeze(1) - new_logps).mean()
                kl_divs.append(kl.item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())
            
            # Early stopping based on KL divergence (averaged over epoch)
            if np.mean(kl_divs[-dataset_size//self.batch_size:]) > self.target_kl * 1.5:
                break
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'kl_divergence': np.mean(kl_divs),
            'training_epochs': i + 1,
            'entropy': np.mean(entropies)
        }