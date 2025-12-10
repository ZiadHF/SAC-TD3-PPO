import gymnasium as gym
import wandb
import torch
import numpy as np
import argparse
import yaml
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Fix Windows encoding issues with emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Import all agents (ensure these exist in your local files)
from algorithms import SACAgent, TD3Agent, PPOAgent
from algorithms import SACAgentCNN, TD3AgentCNN, PPOAgentCNN
from utils.wrappers import PreprocessCarRacing, FrameStack
from gymnasium.wrappers import RecordVideo  # Added for video recording

def make_env(env_id: str, seed: int = 42, use_cnn: bool = False, 
             capture_video: bool = False, run_name: str = None) -> gym.Env:
    """Create environment with optional CNN preprocessing and Video Recording"""
    
    # 1. Initialize with rgb_array to support video recording
    render_mode = 'rgb_array' if capture_video else None
    env = gym.make(env_id, continuous=True, render_mode=render_mode)
    
    # 2. Add Video Recorder (Wrap BEFORE preprocessing to capture clean game)
    if capture_video and run_name:
        video_folder = f"videos/{run_name}"
        # Record every episode (or adjust trigger as needed)
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=lambda x: True, # Record all eval episodes
            disable_logger=True
        )
        print(f"[VIDEO] Recording enabled -> {video_folder}")

    # 3. Apply CNN Preprocessing
    if use_cnn:
        print("[IMG] Applying CNN preprocessing (grayscale + frame stack)")
        env = PreprocessCarRacing(env, resize=(84, 84))
        env = FrameStack(env, num_stack=4)
    
    env.reset(seed=seed)
    return env

def evaluate(agent, env: gym.Env, n_episodes: int = 5, max_ep_len: int = 1000) -> Dict[str, float]:
    """Evaluate agent across multiple episodes"""
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < max_ep_len:
            # Handle both MLP and CNN agents uniformly
            if isinstance(agent, (PPOAgent, PPOAgentCNN)):
                action, _, _ = agent.select_action(obs, deterministic=True)
            else:
                action = agent.select_action(obs, deterministic=True)
            
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_reward += reward
            steps += 1
        
        rewards.append(ep_reward)
        print(f"    Episode {ep+1}: {ep_reward:.2f}")
    
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards)
    }

def create_agent(config: Dict[str, Any], env: gym.Env) -> Any:
    """Factory that creates the correct agent based on config"""
    algo = config['algo']
    use_cnn = config.get('use_cnn', False)
    
    # Parameters that are only for specific algorithms
    ppo_only_params = ['max_ep_len', 'lam', 'clip_ratio', 'train_pi_iters', 'train_v_iters', 'target_kl']
    sac_td3_only_params = ['tau', 'alpha', 'automatic_entropy_tuning', 'buffer_size', 
                          'policy_noise', 'noise_clip', 'policy_delay', 'exploration_noise']
    
    # Determine which params to filter based on algorithm
    is_ppo = 'ppo' in algo.lower()
    if is_ppo:
        filter_params = sac_td3_only_params + ['obs_shape', 'feature_dim']
    else:
        filter_params = ppo_only_params + ['obs_shape', 'feature_dim']
    
    # Base parameters for all agents
    base_params = {
        'action_dim': env.action_space.shape[0],
        'device': config['device'],
        **{k: v for k, v in config['agent_params'].items() 
           if k not in filter_params}
    }
    
    # Select agent class and add environment-specific parameters
    if use_cnn:
        print(f"[AGENT] Creating CNN agent: {algo}")
        
        # All CNN agents need these
        base_params['obs_shape'] = env.observation_space.shape
        base_params['feature_dim'] = config.get('feature_dim', 512)  # Get from top-level config
        
        # Map algorithm to CNN variant
        agent_map = {
            'sac': SACAgentCNN,
            'td3': TD3AgentCNN,
            'ppo': PPOAgentCNN,
            'sac_cnn': SACAgentCNN,
            'td3_cnn': TD3AgentCNN,
            'ppo_cnn': PPOAgentCNN
        }
        
        # Simple lookup
        agent_class = agent_map.get(algo.lower(), None)
    else:
        print(f"[AGENT] Creating MLP agent: {algo}")
        
        # All MLP agents need obs_dim
        base_params['obs_dim'] = env.observation_space.shape[0]
        
        agent_map = {
            'sac': SACAgent,
            'td3': TD3Agent,
            'ppo': PPOAgent
        }
        agent_class = agent_map.get(algo.lower(), None)
    
    if agent_class is None:
        raise ValueError(f"[ERROR] Unknown algorithm: {algo}. Available: {list(agent_map.keys())}")
    
    # Create agent
    try:
        agent = agent_class(**base_params)
        print(f"[OK] Agent created: {agent.__class__.__name__}")
        return agent
    except Exception as e:
        print(f"[ERROR] Failed to create agent: {e}")
        print(f"Parameters: {base_params}")
        raise

def save_model(agent, path, config, eval_score, step):
    """Helper to save model state"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {
        'actor_state_dict': agent.actor.state_dict(),
        'config': config,
        'eval_score': eval_score,
        'step': step
    }
    if hasattr(agent, 'critic'):
        save_dict['critic_state_dict'] = agent.critic.state_dict()
    
    torch.save(save_dict, path)
    print(f"[SAVE] Model saved to {path}")

def load_checkpoint(agent, checkpoint_path):
    """Load checkpoint and return the step number and best eval score"""
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    if hasattr(agent, 'critic') and 'critic_state_dict' in checkpoint:
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"[LOAD] Checkpoint loaded from {checkpoint_path}")
    print(f"       Step: {checkpoint['step']}, Eval Score: {checkpoint['eval_score']:.2f}")
    return checkpoint['step'], checkpoint['eval_score']

def find_latest_checkpoint(run_name):
    """Find the latest checkpoint for a given run"""
    model_dir = f"models/{run_name}"
    if not os.path.exists(model_dir):
        return None
    
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
    if not checkpoints:
        return None
    
    # Extract step numbers and find max
    checkpoint_steps = [(int(f.split('_')[1].split('.')[0]), f) for f in checkpoints]
    latest_step, latest_file = max(checkpoint_steps, key=lambda x: x[0])
    return os.path.join(model_dir, latest_file)

def train(config: Dict[str, Any]) -> float:
    """Main training function"""
    # Validate config
    required_keys = ['algo', 'env_id', 'device', 'total_steps', 'eval_interval', 'eval_episodes']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Initialize wandb
    run_name = config.get('run_name')
    wandb.init(
        project="cmps458-assignment4_2",
        entity="ziadhf-cairo-university",
        config=config,
        name=run_name,
        tags=[config['algo'], config['env_id']],
        monitor_gym=True,
        save_code=True
    )
    
    print(f"\n{'='*70}")
    print(f"[START] Training: {run_name}")
    print(f"{'='*70}")
    
    # Environment setup
    use_cnn = config.get('use_cnn', False)
    
    # Training Env (No video recording to save speed)
    env = make_env(config['env_id'], seed=config['seed'], use_cnn=use_cnn, capture_video=False)
    
    # Eval Env (With video recording)
    eval_env = make_env(config['env_id'], seed=config['seed']+100, use_cnn=use_cnn, 
                        capture_video=True, run_name=run_name)
    
    print(f"[ENV] Environment: {config['env_id']}")
    print(f"   Obs space: {env.observation_space}")
    print(f"   Act space: {env.action_space}")
    print(f"   CNN Mode: {'ON' if use_cnn else 'OFF'}\n")
    
    # Create agent
    agent = create_agent(config, env)
    
    # Training variables
    start_step = 0
    best_eval_score = -np.inf
    
    # Resume from checkpoint if requested
    if config.get('resume', False):
        checkpoint_path = find_latest_checkpoint(run_name)
        if checkpoint_path:
            start_step, best_eval_score = load_checkpoint(agent, checkpoint_path)
            print(f"[RESUME] Continuing from step {start_step}")
        else:
            print(f"[WARNING] No checkpoint found for {run_name}, starting from scratch")
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    eval_interval = config['eval_interval']
    checkpoint_freq = config.get('checkpoint_freq', 100000)
    
    # Enhanced tracking
    episode_count = 0
    episode_rewards_history = []  
    last_log_step = 0
    log_freq = 1000 
    
    # Prefill replay buffer for off-policy agents
    prefill_steps = config.get('learning_starts', 0)
    if prefill_steps > 0 and not isinstance(agent, (PPOAgent, PPOAgentCNN)):
        print(f"[BUFFER] Prefilling buffer ({prefill_steps} steps)...")
        while len(agent.replay_buffer) < prefill_steps:
            action = env.action_space.sample()
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.replay_buffer.add(obs, action, reward, next_obs, term or trunc)
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
        print("[OK] Buffer prefilled\n")
        obs, _ = env.reset() # Reset for actual training
    
    # Main training loop
    for step in range(start_step, config['total_steps']):
        # Action selection
        if isinstance(agent, (PPOAgent, PPOAgentCNN)):
            action, logp, val = agent.select_action(obs, deterministic=False)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.store(obs, action, reward, val, logp, done)
        else:
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
        # Training Logic
        metrics = None
        should_train = False
        if isinstance(agent, (PPOAgent, PPOAgentCNN)):
            should_train = len(agent.buffer.obs) >= config.get('rollout_length', 2048)
        else:
            should_train = step > prefill_steps and step % config['train_freq'] == 0
            
        if should_train:
            if isinstance(agent, (PPOAgent, PPOAgentCNN)):
                metrics = agent.train_step()
            else:
                for _ in range(config['gradient_steps']):
                    metrics = agent.train_step()
            
            # Log training metrics
            if metrics:
                wandb.log({f'train/{k}': v for k, v in metrics.items()}, step=step)
        
        # Periodic Checkpointing (New)
        if step > 0 and step % checkpoint_freq == 0:
            ckpt_path = f"models/{run_name}/checkpoint_{step}.pth"
            save_model(agent, ckpt_path, config, best_eval_score, step)

        # Episode end
        if done or episode_length >= config.get('max_ep_len', 1000):
            episode_count += 1
            episode_rewards_history.append(episode_reward)
            if len(episode_rewards_history) > 100:
                episode_rewards_history.pop(0)
            
            # Reset
            obs, _ = env.reset()
            
            # Determine success
            success = False
            if 'CarRacing' in config['env_id']:
                success = episode_reward >= 900
            
            wandb.log({
                'train/episode_reward': episode_reward,
                'train/episode_length': episode_length,
                'train/episode_count': episode_count,
                'train/success': int(success),
                'train/rolling_mean_reward': np.mean(episode_rewards_history),
            }, step=step)
            
            episode_reward = 0
            episode_length = 0
            
            # Console Progress
            if step - last_log_step >= log_freq:
                progress = 100 * step / config['total_steps']
                recent_mean = np.mean(episode_rewards_history[-10:]) if episode_rewards_history else 0
                print(f"[PROG] {step:,}/{config['total_steps']:,} ({progress:.1f}%) | "
                      f"Ep: {episode_count} | R_Avg: {recent_mean:.1f} | Best: {best_eval_score:.1f}")
                last_log_step = step
            
        # Evaluation & Best Model Saving
        if step % eval_interval == 0 and step > 0:
            print(f"\n[EVAL] Evaluating at step {step:,}...")
            # This triggers the RecordVideo wrapper in eval_env
            eval_results = evaluate(agent, eval_env, n_episodes=config['eval_episodes'])
            wandb.log({f'eval/{k}': v for k, v in eval_results.items()}, step=step)
            
            # Save best model
            if eval_results['mean'] > best_eval_score:
                best_eval_score = eval_results['mean']
                print(f"[BEST] New best score: {best_eval_score:.2f}")
                
                # Save locally
                best_path = f"models/{run_name}/best_model.pth"
                save_model(agent, best_path, config, best_eval_score, step)
                
            obs, _ = env.reset() # Reset training env just in case

    # Final Save
    final_path = f"models/{run_name}/final_model.pth"
    save_model(agent, final_path, config, best_eval_score, step)
    
    print(f"\n{'='*70}")
    print(f"[DONE] Training Complete!")
    print(f"  Best Score: {best_eval_score:.2f}")
    print(f"  Models saved in: models/{run_name}/")
    print(f"{'='*70}\n")
    
    wandb.finish()
    env.close()
    eval_env.close()
    return best_eval_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agents")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    parser.add_argument('--resume', type=str, default=None, help="Run name to resume from (e.g., sac_cnn-CarRacing-v3-20251210_035927)")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add derived fields
        if args.resume:
            config['run_name'] = args.resume
            config['resume'] = True
        else:
            config['run_name'] = f"{config['algo']}-{config['env_id']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            config['resume'] = False
        
        train(config)
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)