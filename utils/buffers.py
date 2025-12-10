import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim):
        self.capacity = capacity
        
        # Optimize memory for images: store as uint8
        # Heuristic: if shape is 3D (H, W, C) or (C, H, W), assume image
        self.is_image = len(obs_shape) == 3
        self.obs_dtype = np.uint8 if self.is_image else np.float32
        
        self.obs = np.zeros((capacity, *obs_shape), dtype=self.obs_dtype)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=self.obs_dtype)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done):
        if self.is_image:
            # Assume input is float [0, 1], convert to uint8 [0, 255]
            # Use np.clip to ensure values are within bounds before casting
            self.obs[self.ptr] = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
            self.next_obs[self.ptr] = (np.clip(next_obs, 0, 1) * 255).astype(np.uint8)
        else:
            self.obs[self.ptr] = obs
            self.next_obs[self.ptr] = next_obs
            
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, batch_size)
        
        obs_batch = self.obs[idxs]
        next_obs_batch = self.next_obs[idxs]
        
        if self.is_image:
            # Convert back to float [0, 1]
            obs_batch = obs_batch.astype(np.float32) / 255.0
            next_obs_batch = next_obs_batch.astype(np.float32) / 255.0
            
        batch = {
            'obs': torch.as_tensor(obs_batch, dtype=torch.float32, device=device),
            'actions': torch.as_tensor(self.actions[idxs], dtype=torch.float32, device=device),
            'rewards': torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=device),
            'next_obs': torch.as_tensor(next_obs_batch, dtype=torch.float32, device=device),
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