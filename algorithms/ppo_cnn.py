import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .ppo import PPOAgent
from utils.networks import ConvGaussianPolicy, NatureCNN, MLP
from utils.buffers import PPORolloutBuffer

class PPOAgentCNN(PPOAgent):
    def __init__(self, obs_shape, action_dim, feature_dim=512, **kwargs):
        # --- TUNED DEFAULTS FOR CARRACING ---
        # These only apply if you DON'T pass them in your config yaml
        kwargs.setdefault('lr', 2.5e-4)
        kwargs.setdefault('gamma', 0.99)
        kwargs.setdefault('clip_ratio', 0.2)
        kwargs.setdefault('lam', 0.95)
        kwargs.setdefault('train_pi_iters', 10) # Lower epochs prevents overfitting
        kwargs.setdefault('train_v_iters', 10)
        kwargs.setdefault('target_kl', 0.03)    # Higher KL tolerance for images
        kwargs.setdefault('ent_coef', 0.01)     # Force exploration (turning)
        kwargs.setdefault('batch_size', 256)    # Larger batch for stable gradients
        
        self.device = kwargs['device']
        self.gamma = kwargs['gamma']
        self.clip_ratio = kwargs['clip_ratio']
        self.lam = kwargs['lam']
        self.train_pi_iters = kwargs['train_pi_iters']
        self.train_v_iters = kwargs['train_v_iters']
        self.target_kl = kwargs['target_kl']
        self.ent_coef = kwargs['ent_coef']
        self.batch_size = kwargs['batch_size']
        self.max_ep_len = kwargs.get('max_ep_len', 1000)
        
        hidden_dims = kwargs.get('hidden_dims', [256])
        
        # Initialize Actor
        self.actor = ConvGaussianPolicy(obs_shape, action_dim, feature_dim, hidden_dims=hidden_dims).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=kwargs['lr'])
        
        # Internal Critic Class to match shapes
        class CriticCNN(nn.Module):
            def __init__(self, obs_shape, feature_dim, hidden_dims):
                super().__init__()
                self.cnn = NatureCNN(obs_shape, feature_dim)
                self.fc = MLP(feature_dim, hidden_dims, 1)
            
            def forward(self, obs):
                return self.fc(self.cnn(obs))
        
        self.critic = CriticCNN(obs_shape, feature_dim, hidden_dims).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=kwargs['lr'])
        
        self.buffer = PPORolloutBuffer()

    def select_action(self, obs, deterministic=False):
        # shape comes in as (C, H, W) numpy array
        # unsqueeze(0) makes it (1, C, H, W) batch
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(obs_tensor, deterministic)
            value = self.critic(obs_tensor)
            
        if log_prob is None: # Deterministic case
            return action.cpu().numpy()[0], 0.0, value.cpu().numpy()[0].item()
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0].item(), value.cpu().numpy()[0].item()
    
    # train_step uses the parent PPOAgent logic or can be pasted here if your parent class 
    # doesn't handle the CNN actor/critic separation correctly. 
    # Since your original code had train_step inside PPOAgent, this inherits fine 
    # AS LONG AS PPOAgent.train_step calls self.actor.evaluate_actions correctly.