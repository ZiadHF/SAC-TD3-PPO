import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# MLP Networks
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU, output_activation=None):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), activation()])
        layers.append(nn.Linear(dims[-1], output_dim))
        if output_activation:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256], log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = MLP(obs_dim, hidden_dims, action_dim * 2)
        self.action_dim = action_dim

    def forward(self, obs):
        output = self.net(obs)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        if deterministic:
            return torch.tanh(mean), None, None
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1)
        
        return action, log_prob, entropy

    def evaluate_actions(self, obs, actions):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Invert tanh to get x_t
        # x_t = atanh(actions)
        # Clamp actions to avoid nan in atanh
        actions = torch.clamp(actions, -0.999999, 0.999999)
        x_t = 0.5 * (torch.log(1 + actions) - torch.log(1 - actions))
        
        log_prob = normal.log_prob(x_t)
        # Tanh correction
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1)
        
        return log_prob, entropy

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))

class TwinCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden_dims)
        self.q2 = Critic(obs_dim, action_dim, hidden_dims)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)

    def min_q(self, obs, action):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)

# CNN Networks
class NatureCNN(nn.Module):
    def __init__(self, obs_shape, feature_dim=512):
        super().__init__()
        # obs_shape from gym: (H, W, C) -> convert to (C, H, W) for PyTorch conv
        if len(obs_shape) == 3:
            # Assume gym format (H, W, C) if last dim is small (channels)
            if obs_shape[2] <= 4:
                self.input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
            else:
                self.input_shape = obs_shape  # Already (C, H, W)
        else:
            self.input_shape = obs_shape
            
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            dummy = self.conv3(self.conv2(self.conv1(dummy)))
            conv_out_size = dummy.numel()
        
        self.fc = nn.Linear(conv_out_size, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        # Check if input is (B, H, W, C) and needs permutation to (B, C, H, W)
        # We use self.input_shape[0] which stores the expected channel count
        if len(x.shape) == 4:
            if x.shape[1] != self.input_shape[0] and x.shape[-1] == self.input_shape[0]:
                x = x.permute(0, 3, 1, 2).contiguous()
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))

class ConvGaussianPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dims=[256, 256], log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.cnn = NatureCNN(obs_shape, feature_dim)
        self.mean_net = MLP(feature_dim, hidden_dims, action_dim)
        self.log_std_net = MLP(feature_dim, hidden_dims, action_dim)
        self.action_dim = action_dim

    def forward(self, obs):
        features = self.cnn(obs)
        mean = self.mean_net(features)
        log_std = torch.clamp(self.log_std_net(features), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        if deterministic:
            return torch.tanh(mean), None, None
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1)
        
        return action, log_prob, entropy

    def evaluate_actions(self, obs, actions):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Invert tanh to get x_t
        actions = torch.clamp(actions, -0.999999, 0.999999)
        x_t = 0.5 * (torch.log(1 + actions) - torch.log(1 - actions))
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1)
        
        return log_prob, entropy

class ConvCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dims=[256, 256]):
        super().__init__()
        self.cnn = NatureCNN(obs_shape, feature_dim)
        self.net = MLP(feature_dim + action_dim, hidden_dims, 1)

    def forward(self, obs, action):
        features = self.cnn(obs)
        return self.net(torch.cat([features, action], dim=-1))

class ConvTwinCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dims=[256, 256]):
        super().__init__()
        self.q1 = ConvCritic(obs_shape, action_dim, feature_dim, hidden_dims)
        self.q2 = ConvCritic(obs_shape, action_dim, feature_dim, hidden_dims)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)

    def min_q(self, obs, action):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)