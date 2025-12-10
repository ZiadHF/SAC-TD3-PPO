import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .sac import SACAgent
from utils.networks import ConvGaussianPolicy, ConvTwinCritic
from utils.buffers import ReplayBuffer

class SACAgentCNN(SACAgent):
    def __init__(self, obs_shape, action_dim, feature_dim=512, **kwargs):
        # Initialize parameters exactly like parent
        self.device = kwargs['device']
        self.gamma = kwargs['gamma']
        self.tau = kwargs['tau']
        self.alpha = kwargs.get('alpha', 0.2)
        self.batch_size = kwargs['batch_size']
        
        hidden_dims = kwargs.get('hidden_dims', [256, 256])
        
        # --- CNN Networks ---
        self.actor = ConvGaussianPolicy(obs_shape, action_dim, feature_dim, hidden_dims=hidden_dims).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=kwargs['lr'])
        
        self.critic = ConvTwinCritic(obs_shape, action_dim, feature_dim, hidden_dims=hidden_dims).to(self.device)
        self.critic_target = ConvTwinCritic(obs_shape, action_dim, feature_dim, hidden_dims=hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=kwargs['lr'])
        
        self.automatic_entropy_tuning = kwargs.get('automatic_entropy_tuning', True)
        if self.automatic_entropy_tuning:
            self.target_entropy = kwargs.get('target_entropy', -action_dim)
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=kwargs['lr'])
            self.alpha = self.log_alpha.exp()
        
        # Buffer initialization
        self.replay_buffer = ReplayBuffer(kwargs['buffer_size'], obs_shape, action_dim)

    def select_action(self, obs, deterministic=False):
        # Convert obs to tensor and add batch dimension
        # Obs comes from FrameStack wrapper in (H, W, C) format, already normalized [0, 1]
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        # Add batch dimension if needed
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # Note: No permutation here - the CNN network handles HWC→CHW conversion internally

        with torch.no_grad():
            action, _ = self.actor.sample(obs_tensor, deterministic)
        return action.cpu().numpy()[0]

    def train_step(self):
        # Standard check
        if self.replay_buffer.size < self.batch_size:
            return None
            
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        
        # --- CNN SPECIFIC PREPROCESSING ---
        # Observations are already normalized [0, 1] from FrameStack wrapper
        obs = batch['obs'].float()
        next_obs = batch['next_obs'].float()

        # Note: No permutation here - the CNN network handles HWC→CHW conversion internally

        # Scale Rewards: CarRacing rewards are huge (~900), scale to prevent value explosion
        rewards = batch['rewards'] / 20.0 
        # ----------------------------------

        # --- STANDARD SAC LOGIC (using processed tensors) ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            # Use 'min_q' if ConvTwinCritic supports it, otherwise manually compute min(q1, q2)
            if hasattr(self.critic_target, 'min_q'):
                min_q_next = self.critic_target.min_q(next_obs, next_action)
            else:
                q1_next, q2_next = self.critic_target(next_obs, next_action)
                min_q_next = torch.min(q1_next, q2_next)
                
            q_target = rewards.unsqueeze(1) + (1 - batch['dones'].unsqueeze(1)) * self.gamma * (min_q_next - self.alpha * next_log_prob)
        
        # Critic Update
        q1, q2 = self.critic(obs, batch['actions'])
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optim.step()
        
        # Actor Update
        action_pred, log_prob = self.actor.sample(obs)
        
        if hasattr(self.critic, 'min_q'):
            q_new = self.critic.min_q(obs, action_pred)
        else:
            q1_new, q2_new = self.critic(obs, action_pred)
            q_new = torch.min(q1_new, q2_new)
            
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optim.step()
        
        # Entropy Update
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        
        # Soft Update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'entropy': -log_prob.mean().item()
        }