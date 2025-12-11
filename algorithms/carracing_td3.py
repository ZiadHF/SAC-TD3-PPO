"""
CarRacing-v3 TD3 Training with State-of-the-Art Implementation
Optimized for Colab T4 GPU with careful hyperparameter tuning
"""

# ============================================================================
# INSTALLATION & IMPORTS
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
import cv2
import wandb
from typing import Tuple, List
import os
from datetime import datetime

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# ENVIRONMENT WRAPPERS
# ============================================================================

class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB observation to grayscale"""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
    
    def observation(self, obs):
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


class ResizeObservation(gym.ObservationWrapper):
    """Resize observation to specified shape"""
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
    
    def observation(self, obs):
        return cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)


class FrameStack(gym.Wrapper):
    """Stack last n frames"""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        low = np.repeat(self.observation_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], n_frames, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        return np.array(self.frames, dtype=self.observation_space.dtype)


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observation to [0, 1]"""
    def __init__(self, env):
        super().__init__(env)
        low = self.observation_space.low / 255.0
        high = self.observation_space.high / 255.0
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
    
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class RewardShaping(gym.Wrapper):
    """Apply reward shaping for better learning"""
    def __init__(self, env):
        super().__init__(env)
        self.neg_reward_counter = 0
        self.tile_visited_count = 0
        
    def reset(self, **kwargs):
        self.neg_reward_counter = 0
        self.tile_visited_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Penalize negative rewards (going off-track)
        if reward < 0:
            self.neg_reward_counter += 1
            reward -= 0.1
        else:
            self.neg_reward_counter = 0
        
        # Early termination if too many negative rewards
        if self.neg_reward_counter > 30:
            terminated = True
            reward -= 10.0
        
        return obs, reward, terminated, truncated, info


class ActionRepeat(gym.Wrapper):
    """Repeat actions for better stability"""
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env():
    """Create wrapped CarRacing environment"""
    env = gym.make('CarRacing-v3', continuous=True, render_mode=None)
    env = ActionRepeat(env, repeat=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStack(env, n_frames=4)
    env = NormalizeObservation(env)
    env = RewardShaping(env)
    return env


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class CNNFeatureExtractor(nn.Module):
    """CNN for extracting features from stacked frames"""
    def __init__(self, n_frames=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate output size
        with torch.no_grad():
            sample = torch.zeros(1, n_frames, 84, 84)
            self.feature_dim = self.conv(sample).shape[1]
    
    def forward(self, x):
        return self.conv(x)


class Actor(nn.Module):
    """Actor network for TD3"""
    def __init__(self, n_frames=4, action_dim=3):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(n_frames)
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        features = self.feature_extractor(state)
        return self.fc(features)


class Critic(nn.Module):
    """Twin Q-network for TD3"""
    def __init__(self, n_frames=4, action_dim=3):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(n_frames)
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        features = self.feature_extractor(state)
        sa = torch.cat([features, action], dim=1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state, action):
        features = self.feature_extractor(state)
        sa = torch.cat([features, action], dim=1)
        return self.q1(sa)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer with prioritization support"""
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# TD3 AGENT
# ============================================================================

class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient Agent"""
    def __init__(
        self,
        n_frames=4,
        action_dim=3,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        exploration_noise=0.1
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        
        # Networks
        self.actor = Actor(n_frames, action_dim).to(device)
        self.actor_target = Actor(n_frames, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(n_frames, action_dim).to(device)
        self.critic_target = Critic(n_frames, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.total_it = 0
    
    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if explore:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def train(self, replay_buffer, batch_size=128):
        self.total_it += 1
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        with torch.no_grad():
            # Select action with target policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Update critics
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Delayed policy updates
        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            # Update actor
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else 0.0,
            'q_value': current_q1.mean().item()
        }
    
    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_td3(
    total_timesteps=500000,
    batch_size=128,
    buffer_size=100000,
    learning_starts=10000,
    save_freq=50000,
    eval_freq=10000,
    eval_episodes=5,
    project_name="carracing-td3",
    run_name=None
):
    """Main training loop"""
    
    # Initialize wandb
    if run_name is None:
        run_name = f"td3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "algorithm": "TD3",
            "env": "CarRacing-v3",
            "total_timesteps": total_timesteps,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
        }
    )
    
    # Create environment and agent
    env = make_env()
    eval_env = make_env()
    agent = TD3Agent(
        n_frames=4,
        action_dim=3,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        exploration_noise=0.1
    )
    replay_buffer = ReplayBuffer(buffer_size)
    
    # Training variables
    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Using device: {device}")
    
    for t in range(total_timesteps):
        episode_timesteps += 1
        
        # Select action
        if t < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, explore=True)
        
        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        # Train agent
        if t >= learning_starts:
            train_metrics = agent.train(replay_buffer, batch_size)
            
            if t % 1000 == 0:
                wandb.log({
                    **train_metrics,
                    'timestep': t
                })
        
        # End of episode
        if done:
            print(f"Episode {episode_num + 1} | Timestep {t + 1} | Reward: {episode_reward:.2f} | Steps: {episode_timesteps}")
            
            wandb.log({
                'episode_reward': episode_reward,
                'episode_length': episode_timesteps,
                'episode': episode_num,
                'timestep': t
            })
            
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Evaluation
        if (t + 1) % eval_freq == 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                eval_state, _ = eval_env.reset()
                eval_reward = 0
                eval_done = False
                
                while not eval_done:
                    eval_action = agent.select_action(eval_state, explore=False)
                    eval_state, reward, terminated, truncated, _ = eval_env.step(eval_action)
                    eval_reward += reward
                    eval_done = terminated or truncated
                
                eval_rewards.append(eval_reward)
            
            avg_eval_reward = np.mean(eval_rewards)
            print(f"\n{'='*50}")
            print(f"Evaluation at timestep {t + 1}")
            print(f"Average reward: {avg_eval_reward:.2f} ± {np.std(eval_rewards):.2f}")
            print(f"{'='*50}\n")
            
            wandb.log({
                'eval/mean_reward': avg_eval_reward,
                'eval/std_reward': np.std(eval_rewards),
                'timestep': t
            })
        
        # Save model
        if (t + 1) % save_freq == 0:
            save_path = f"checkpoints/td3_carracing_{t+1}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            agent.save(save_path)
            print(f"Model saved to {save_path}")
    
    # Final save
    agent.save("checkpoints/td3_carracing_final.pt")
    print("Training complete! Final model saved.")
    
    env.close()
    eval_env.close()
    wandb.finish()
    
    return agent

# Add this function after the train_td3 function and before if __name__ == "__main__":

def record_episodes(model_path: str, output_dir: str = "recordings", num_episodes: int = 5):
    """
    Load a trained TD3 model and record episodes using gym's RecordVideo wrapper
    
    Args:
        model_path: Path to the .pt checkpoint file
        output_dir: Directory to save video recordings
        num_episodes: Number of episodes to record
    """
    import os
    from gymnasium.wrappers import RecordVideo
    
    print(f"Loading model from {model_path}...")
    
    # Create agent
    agent = TD3Agent(
        n_frames=4,
        action_dim=3,
        lr_actor=1e-4,
        lr_critic=1e-3,
    )
    
    # Load trained weights
    agent.load(model_path)
    agent.actor.eval()  # Set to evaluation mode
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment with video recording
    # First create base environment with render_mode
    base_env = gym.make('CarRacing-v3', continuous=True, render_mode='rgb_array')
    
    # Wrap with RecordVideo before other wrappers
    env = RecordVideo(
        base_env,
        video_folder=output_dir,
        episode_trigger=lambda episode_id: True,  # Record all episodes
        name_prefix="td3_carracing"
    )
    
    # Apply preprocessing wrappers
    env = ActionRepeat(env, repeat=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStack(env, n_frames=4)
    env = NormalizeObservation(env)
    
    print(f"Recording {num_episodes} episodes to {output_dir}...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Select action without exploration noise
            action = agent.select_action(state, explore=False)
            
            # Step environment
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f} - Steps: {steps}")
    
    env.close()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Recording Summary:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Videos saved to: {output_dir}")
    print(f"{'='*50}\n")
    
    return episode_rewards


# Example usage in the main block:
if __name__ == "__main__":
    # # Login to wandb (you'll need to enter your API key)
    # wandb.login()
    
    # # Train the agent
    # agent = train_td3(
    #     total_timesteps=500000,
    #     batch_size=128,
    #     buffer_size=100000,
    #     learning_starts=10000,
    #     save_freq=50000,
    #     eval_freq=10000,
    #     eval_episodes=5,
    #     project_name="carracing-td3"
    # )
    
    # print("Training completed successfully!")
    # print("Saved models can be found in the 'checkpoints' directory")
    
    # Uncomment to record episodes after training:
    record_episodes("td3_carracing_50000.pt", output_dir="recordings", num_episodes=5)
