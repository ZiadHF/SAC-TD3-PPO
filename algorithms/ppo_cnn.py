import torch
import torch.optim as optim
import torch.nn.functional as F
from .ppo import PPOAgent
from utils.networks import ConvGaussianPolicy, NatureCNN, MLP
from utils.buffers import PPORolloutBuffer

class PPOAgentCNN(PPOAgent):
    def __init__(self, obs_shape, action_dim, feature_dim=512, **kwargs):
        self.device = kwargs['device']
        self.gamma = kwargs['gamma']
        self.clip_ratio = kwargs['clip_ratio']
        self.lam = kwargs['lam']
        self.train_pi_iters = kwargs['train_pi_iters']
        self.train_v_iters = kwargs['train_v_iters']
        self.target_kl = kwargs['target_kl']
        self.max_ep_len = kwargs['max_ep_len']
        self.ent_coef = kwargs.get('ent_coef', 0.0)
        self.batch_size = kwargs.get('batch_size', 64)
        
        # Extract only the hidden_dims for networks
        hidden_dims = kwargs.get('hidden_dims', [256, 256])
        
        # CNN actor
        self.actor = ConvGaussianPolicy(obs_shape, action_dim, feature_dim, hidden_dims=hidden_dims).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=kwargs['lr'])
        
        # CNN critic
        class CriticCNN(torch.nn.Module):
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
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.sample(obs_tensor, deterministic)
            value = self.critic(obs_tensor)
        # log_prob is None when deterministic=True
        if log_prob is None:
            return action.cpu().numpy()[0], 0.0, value.cpu().numpy()[0].item()
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0].item(), value.cpu().numpy()[0].item()