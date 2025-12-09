import torch
import torch.optim as optim
import torch.nn.functional as F
from .sac import SACAgent
from utils.networks import ConvGaussianPolicy, ConvTwinCritic
from utils.buffers import ReplayBuffer

class SACAgentCNN(SACAgent):
    def __init__(self, obs_shape, action_dim, feature_dim=512, **kwargs):
        self.device = kwargs['device']
        self.gamma = kwargs['gamma']
        self.tau = kwargs['tau']
        self.alpha = kwargs.get('alpha', 0.2)
        self.batch_size = kwargs['batch_size']
        
        # Extract only the hidden_dims for networks
        hidden_dims = kwargs.get('hidden_dims', [256, 256])
        
        # CNN networks
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
        
        self.replay_buffer = ReplayBuffer(kwargs['buffer_size'], obs_shape, action_dim)

    def select_action(self, obs, deterministic=False):
        # obs: (H, W, C) -> add batch dim
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_tensor, deterministic)
        return action.cpu().numpy()[0]