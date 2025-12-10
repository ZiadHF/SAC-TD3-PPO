import gymnasium as gym
import wandb
import torch
import numpy as np
import argparse
import yaml
import os
import sys
from datetime import datetime
from typing import Dict, Any, Tuple

import random

# Fix Windows encoding issues with emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Import all agents
from algorithms import SACAgent, TD3Agent, PPOAgent
from algorithms import SACAgentCNN, TD3AgentCNN, PPOAgentCNN
from utils.wrappers import PreprocessCarRacing, FrameStack, RepeatAction
from gymnasium.wrappers import RecordVideo

def set_seed(seed: int):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"[SEED] Set global seed to {seed}")

def make_env(env_id: str, seed: int = 42, use_cnn: bool = False, render_mode: str = None, video_folder: str = None, video_interval: int = 10) -> gym.Env:
    """Create environment with optional CNN preprocessing and video recording"""
    # Increase max_episode_steps for CarRacing to account for frame skipping
    # Default is 1000. With skip=4, we need 4000 to maintain same duration.
    max_steps = 4000 if 'CarRacing' in env_id and use_cnn else None
    
    if max_steps:
        env = gym.make(env_id, continuous=True, max_episode_steps=max_steps, render_mode=render_mode)
    else:
        env = gym.make(env_id, continuous=True, render_mode=render_mode)
    
    # Add video recording if requested
    if video_folder:
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=lambda x: x % video_interval == 0, # Record first episode of every eval batch
            disable_logger=True
        )
    
    if use_cnn:
        print(f"[IMG] Applying CNN preprocessing (grayscale + skip + stack) | Max Steps: {max_steps}")
        env = PreprocessCarRacing(env, resize=(84, 84))
        env = RepeatAction(env, skip=4)
        env = FrameStack(env, num_stack=4)
    
    env.reset(seed=seed)
    return env

def save_model(agent, config: Dict[str, Any], score: float, step: int, suffix: str = "best") -> str:
    """Save model checkpoint"""
    run_name = config.get('run_name', f"{config['algo']}-{config['env_id']}")
    save_path = f"models/{run_name}_{suffix}.pth"
    os.makedirs("models", exist_ok=True)
    
    save_dict = {
        'actor_state_dict': agent.actor.state_dict(),
        'config': config,
        'eval_score': score,
        'step': step
    }
    if hasattr(agent, 'critic'):
        save_dict['critic_state_dict'] = agent.critic.state_dict()
    
    torch.save(save_dict, save_path)
    print(f"[{suffix.upper()}] Saved model to {save_path}")
    return save_path

def load_model(agent, checkpoint_path: str):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print(f"[LOAD] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    
    # Load actor
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    
    # Load critic if available
    if hasattr(agent, 'critic') and 'critic_state_dict' in checkpoint:
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        # Also load target networks if they exist
        if hasattr(agent, 'critic_target'):
            agent.critic_target.load_state_dict(checkpoint['critic_state_dict'])
    
    if hasattr(agent, 'actor_target') and 'actor_state_dict' in checkpoint:
        agent.actor_target.load_state_dict(checkpoint['actor_state_dict'])
        
    print(f"[OK] Model loaded successfully (Step: {checkpoint.get('step', 'Unknown')}, Score: {checkpoint.get('eval_score', 'Unknown')})")
    return checkpoint

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
        base_params['feature_dim'] = config['feature_dim']
        
        # Map algorithm to CNN variant
        agent_map = {
            'sac': SACAgentCNN,
            'td3': TD3AgentCNN,
            'ppo': PPOAgentCNN,
            'sac_cnn': SACAgentCNN,  # Allow explicit _cnn suffix
            'td3_cnn': TD3AgentCNN,
            'ppo_cnn': PPOAgentCNN
        }
        
        # Remove _cnn suffix from lookup
        agent_class = agent_map.get(algo.replace('_cnn', ''), None)
    else:
        print(f"[AGENT] Creating MLP agent: {algo}")
        
        # All MLP agents need obs_dim
        base_params['obs_dim'] = env.observation_space.shape[0]
        
        # Standard MLP agents
        agent_map = {
            'sac': SACAgent,
            'td3': TD3Agent,
            'ppo': PPOAgent
        }
        agent_class = agent_map.get(algo, None)
    
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

def train(config: Dict[str, Any], checkpoint_path: str = None) -> float:
    """Main training function"""
    # Validate config
    required_keys = ['algo', 'env_id', 'device', 'total_steps', 'eval_interval', 'eval_episodes']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Initialize wandb
    run_name = f"{config['algo']}-{config['env_id']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="cmps458-assignment4_3",
        config=config,
        entity="ziadhf-cairo-university",
        name=run_name,
        tags=[config['algo'], config['env_id']],
        monitor_gym=True,
        save_code=True
    )
    
    print(f"\n{'='*70}")
    print(f"[START] Training: {run_name}")
    print(f"{'='*70}")
    
    # Display HuggingFace status
    if config.get('publish_to_hub', False):
        if os.getenv('HF_TOKEN'):
            print(f"[HF] HuggingFace uploads enabled to: {config.get('hf_repo_id', 'N/A')}")
        else:
            print(f"[HF] publish_to_hub=true but HF_TOKEN not set")
            print(f"[HF] Models will only be saved locally")
            print(f"[HF] To enable uploads: export HF_TOKEN='your_token_here'")
    else:
        print(f"[HF] HuggingFace uploads disabled (publish_to_hub=false)")
    print(f"[SAVE] Models will be saved to: models/{run_name}_*.pth")
    print()
    
    # Set global seed
    set_seed(config['seed'])
    
    # Environment setup
    use_cnn = config.get('use_cnn', False)
    env = make_env(config['env_id'], seed=config['seed'], use_cnn=use_cnn)
    
    # Eval env with video recording
    video_folder = f"videos/{run_name}"
    eval_env = make_env(
        config['env_id'], 
        seed=config['seed']+100, 
        use_cnn=use_cnn, 
        render_mode='rgb_array',
        video_folder=video_folder,
        video_interval=config['eval_episodes']
    )
    
    print(f"[ENV] Environment: {config['env_id']}")
    print(f"   Obs space: {env.observation_space}")
    print(f"   Act space: {env.action_space}")
    print(f"   CNN Mode: {'ON' if use_cnn else 'OFF'}\n")
    
    # Create agent
    agent = create_agent(config, env)
    
    # Load checkpoint if provided
    start_step = 0
    if checkpoint_path:
        checkpoint = load_model(agent, checkpoint_path)
        start_step = checkpoint.get('step', 0)
        print(f"[RESUME] Resuming training from step {start_step}")
    
    # Training variables
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    best_eval_score = -np.inf
    eval_interval = config['eval_interval']
    
    # Enhanced tracking
    episode_count = 0
    episode_rewards_history = []  # Track recent episode rewards
    total_train_steps = 0
    last_log_step = 0
    log_freq = 1000  # Log training progress every N steps
    
    # Prefill replay buffer for off-policy agents
    prefill_steps = config.get('learning_starts', 0)
    if prefill_steps > 0 and not isinstance(agent, (PPOAgent, PPOAgentCNN)):
        print(f"[BUFFER] Prefilling buffer ({prefill_steps} steps)...")
        for _ in range(prefill_steps):
            action = env.action_space.sample()
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.replay_buffer.add(obs, action, reward, next_obs, term or trunc)
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
        print("[OK] Buffer prefilled\n")
    
    # Main training loop
    try:
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
            
            # Training
            should_train = False
            if isinstance(agent, (PPOAgent, PPOAgentCNN)):
                should_train = len(agent.buffer.obs) >= config.get('rollout_length', 2048)
            else:
                should_train = step > prefill_steps and step % config['train_freq'] == 0
                
            if should_train:
                if isinstance(agent, (PPOAgent, PPOAgentCNN)):
                    metrics = agent.train_step()
                else:
                    metrics = None
                    for _ in range(config['gradient_steps']):
                        metrics = agent.train_step()
                        if metrics:
                            wandb.log({f'train/{k}': v for k, v in metrics.items()}, step=step)
                
                if metrics:
                    wandb.log({f'train/{k}': v for k, v in metrics.items()}, step=step)
            
            # Episode end
            if done or episode_length >= config['max_ep_len']:
                episode_count += 1
                episode_rewards_history.append(episode_reward)
                
                # Keep only recent 100 episodes for rolling average
                if len(episode_rewards_history) > 100:
                    episode_rewards_history.pop(0)
                
                # Determine success (environment-specific thresholds)
                success = False
                if 'LunarLander' in config['env_id']:
                    success = episode_reward >= 200  # Solved threshold
                elif 'CarRacing' in config['env_id']:
                    success = episode_reward >= 900  # Good performance
                
                wandb.log({
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_length,
                    'train/episode_count': episode_count,
                    'train/success': int(success),
                    'train/rolling_mean_reward': np.mean(episode_rewards_history),
                    'train/rolling_std_reward': np.std(episode_rewards_history) if len(episode_rewards_history) > 1 else 0,
                }, step=step)
                
                # Progress logging
                if step - last_log_step >= log_freq:
                    progress = 100 * step / config['total_steps']
                    recent_mean = np.mean(episode_rewards_history[-10:]) if episode_rewards_history else 0
                    print(f"[PROGRESS] Step {step:,}/{config['total_steps']:,} ({progress:.1f}%) | "
                          f"Episodes: {episode_count} | Recent Avg: {recent_mean:.1f} | Best: {best_eval_score:.1f}")
                    last_log_step = step
            
            # Evaluation
            if step % eval_interval == 0 and step > 0:
                print(f"\n[EVAL] Evaluating at step {step:,}...")
                eval_results = evaluate(agent, eval_env, n_episodes=config['eval_episodes'], max_ep_len=config['max_ep_len'])
                wandb.log({f'eval/{k}': v for k, v in eval_results.items()}, step=step)
                
                # Save best model
                if eval_results['mean'] > best_eval_score:
                    best_eval_score = eval_results['mean']
                    print(f"[BEST] New best: {best_eval_score:.2f}")
                    
                    # Always save locally first
                    save_path = save_model(agent, config, best_eval_score, step, suffix="best")
                    
                    # Optional: Upload to HuggingFace Hub (requires HF_TOKEN environment variable)
                    if config.get('publish_to_hub', False) and os.getenv('HF_TOKEN'):
                        try:
                            from huggingface_hub import upload_file
                            upload_file(
                                path_or_fileobj=save_path,
                                path_in_repo="model.pth",
                                repo_id=config.get('hf_repo_id', 'username/model-name'),
                                commit_message=f"Best model: {best_eval_score:.2f}",
                                token=os.getenv('HF_TOKEN')
                            )
                            print(f"[HF] ✓ Uploaded to HuggingFace: {config['hf_repo_id']}")
                        except Exception as e:
                            print(f"[HF] Upload failed: {e}")
                            print(f"[HF] Model saved locally at: {save_path}")
                    elif config.get('publish_to_hub', False) and not os.getenv('HF_TOKEN'):
                        print(f"[HF] publish_to_hub=true but HF_TOKEN not set")
                        print(f"[HF] Model saved locally at: {save_path}")
                        print(f"[HF] Set HF_TOKEN environment variable to enable uploads")
                
                # Save latest checkpoint
                save_model(agent, config, eval_results['mean'], step, suffix="latest")
                
                # Periodic checkpoint (every 5 evals)
                if (step // eval_interval) % 5 == 0:
                    save_model(agent, config, eval_results['mean'], step, suffix=f"step_{step}")
            
            if done or episode_length >= config['max_ep_len']:
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0    
    except KeyboardInterrupt:
        print("\n[STOP] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Final evaluation
        print("\n[EVAL] Running final evaluation...")
        try:
            final_eval = evaluate(agent, eval_env, n_episodes=config['eval_episodes'], max_ep_len=config['max_ep_len'])
            final_score = final_eval['mean']
            print(f"[EVAL] Final Score: {final_score:.2f}")
            wandb.log({f'eval/{k}': v for k, v in final_eval.items()}, step=step if 'step' in locals() else 0)
        except Exception as e:
            print(f"[WARN] Final evaluation failed: {e}")
            final_score = np.mean(episode_rewards_history[-10:]) if episode_rewards_history else 0.0

        # Save final model
        print("\n[SAVE] Saving final model...")
        save_model(agent, config, final_score, step if 'step' in locals() else 0, suffix="final")
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"[DONE] Training Complete/Stopped!")
        print(f"{'='*70}")
        print(f"  Total Steps: {step if 'step' in locals() else 0:,}")
        print(f"  Total Episodes: {episode_count}")
        print(f"  Best Eval Score: {best_eval_score:.2f}")
        if episode_rewards_history:
            print(f"  Final Rolling Avg (100 ep): {np.mean(episode_rewards_history):.2f}")
        print(f"{'='*70}\n")
        
        wandb.finish()
        return best_eval_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agents for CMPS458 Assignment 4")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    parser.add_argument('--load', type=str, default=None, help="Path to model checkpoint to load")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate config
        if 'algo' not in config or 'env_id' not in config:
            raise ValueError("Config must contain 'algo' and 'env_id'")
        
        # Add derived fields
        config['run_name'] = f"{config['algo']}-{config['env_id']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        train(config, checkpoint_path=args.load)
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)