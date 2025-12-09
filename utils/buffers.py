import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, batch_size)
        batch = {
            'obs': torch.as_tensor(self.obs[idxs], dtype=torch.float32, device=device),
            'actions': torch.as_tensor(self.actions[idxs], dtype=torch.float32, device=device),
            'rewards': torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=device),
            'next_obs': torch.as_tensor(self.next_obs[idxs], dtype=torch.float32, device=device),
            'dones': torch.as_tensor(self.dones[idxs], dtype=torch.float32, device=device)
        }
        return batch

class PPORolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logps = []
        self.dones = []

    def store(self, obs, action, reward, value, logp, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.logps.append(logp)
        self.dones.append(done)

    def get(self):
        data = {
            'obs': np.array(self.obs, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
            'logps': np.array(self.logps, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.bool_)
        }
        self.__init__()
        return data