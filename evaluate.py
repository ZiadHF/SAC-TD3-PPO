import torch
import gymnasium as gym
import numpy as np
import os
import wandb
from datetime import datetime
from algorithms import *
from utils.wrappers import PreprocessCarRacing, FrameStack
import argparse

def load_agent(config_path, model_path):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config.get('device', 'cuda')
    
    # Create env
    env = gym.make(config['env_id'], continuous=True)
    if config.get('use_cnn', False):
        env = PreprocessCarRacing(env)
        env = FrameStack(env, num_stack=4)
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        
        if 'sac' in config['algo']:
            agent = SACAgentCNN(obs_shape=obs_shape, action_dim=action_dim, 
                               feature_dim=config.get('feature_dim', 512),
                               device=device, **config['agent_params'])
        elif 'td3' in config['algo']:
            agent = TD3AgentCNN(obs_shape=obs_shape, action_dim=action_dim,
                               feature_dim=config.get('feature_dim', 512),
                               device=device, **config['agent_params'])
        elif 'ppo' in config['algo']:
            agent = PPOAgentCNN(obs_shape=obs_shape, action_dim=action_dim,
                               feature_dim=config.get('feature_dim', 512),
                               device=device, **config['agent_params'])
    else:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        if config['algo'] == 'sac':
            agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, device=device, **config['agent_params'])
        elif config['algo'] == 'td3':
            agent = TD3Agent(obs_dim=obs_dim, action_dim=action_dim, device=device, **config['agent_params'])
        elif config['algo'] == 'ppo':
            agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device, **config['agent_params'])
    
    env.close()
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    if hasattr(agent, 'critic') and 'critic_state_dict' in checkpoint:
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor.eval()
    if hasattr(agent, 'critic'):
        agent.critic.eval()
    
    return agent, config, checkpoint.get('eval_score', 'N/A')

def evaluate_agent(config_path, model_path, n_episodes=10, record_video=False, wandb_log=False):
    agent, config, saved_score = load_agent(config_path, model_path)
    
    print(f"\n{'='*60}")
    print(f"[EVAL] {config['algo']} on {config['env_id']}")
    print(f"[INFO] Saved score: {saved_score}")
    print(f"{'='*60}\n")
    
    # Create evaluation environment
    env = gym.make(config['env_id'], continuous=True, render_mode="rgb_array" if record_video else None)
    if config.get('use_cnn', False):
        env = PreprocessCarRacing(env)
        env = FrameStack(env, num_stack=4)
    
    # Wrap with video recorder if needed
    video_folder = None
    if record_video:
        os.makedirs("videos", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_folder = f"videos/{config['algo']}_{config['env_id'].replace('-', '_')}_{timestamp}"
        
        # For CNN envs, we need a separate recording env
        if config.get('use_cnn', False):
            record_env = gym.make(config['env_id'], continuous=True, render_mode="rgb_array")
            record_env = gym.wrappers.RecordVideo(record_env, video_folder, episode_trigger=lambda x: True)
        else:
            env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)
            record_env = None
    else:
        record_env = None
    
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        if record_env is not None:
            record_env.reset()
            
        done = False
        total = 0
        steps = 0
        
        while not done and steps < 1000:
            if isinstance(agent, (PPOAgent, PPOAgentCNN)):
                action, _, _ = agent.select_action(obs, deterministic=True)
            else:
                action = agent.select_action(obs, deterministic=True)
            
            obs, reward, term, trunc, _ = env.step(action)
            if record_env is not None:
                record_env.step(action)
                
            done = term or trunc
            total += reward
            steps += 1
            
        rewards.append(total)
        lengths.append(steps)
        print(f"  Episode {ep+1}/{n_episodes}: Reward={total:.2f}, Steps={steps}")
    
    env.close()
    if record_env is not None:
        record_env.close()
    
    results = {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'median': np.median(rewards),
        'mean_length': np.mean(lengths)
    }
    
    print(f"\n{'='*60}")
    print(f"[RESULTS]")
    print(f"  Mean:   {results['mean']:.2f} +/- {results['std']:.2f}")
    print(f"  Median: {results['median']:.2f}")
    print(f"  Range:  [{results['min']:.2f}, {results['max']:.2f}]")
    print(f"  Avg Length: {results['mean_length']:.1f} steps")
    print(f"{'='*60}")
    
    if record_video:
        print(f"\n[VIDEO] Saved to: {video_folder}")
    
    # Log to wandb
    if wandb_log:
        run = wandb.init(
            project="cmps458-assignment4",
            name=f"eval-{config['algo']}-{config['env_id']}",
            config=config,
            tags=[config['algo'], config['env_id'], 'evaluation']
        )
        
        wandb.log({
            'eval/mean_reward': results['mean'],
            'eval/std_reward': results['std'],
            'eval/min_reward': results['min'],
            'eval/max_reward': results['max'],
            'eval/median_reward': results['median'],
        })
        
        # Upload videos if recorded
        if record_video and video_folder:
            import glob
            for video_file in glob.glob(f"{video_folder}/*.mp4"):
                wandb.log({"video": wandb.Video(video_file, fps=30, format="mp4")})
                print(f"[WANDB] Uploaded: {video_file}")
        
        wandb.finish()
        print("[OK] Results logged to Wandb")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and record trained RL agents")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--model', type=str, required=True, help="Path to model .pth file")
    parser.add_argument('--episodes', type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument('--record', action='store_true', help="Record video of agent")
    parser.add_argument('--wandb', action='store_true', help="Log results to Wandb")
    args = parser.parse_args()
    
    results = evaluate_agent(
        args.config, 
        args.model, 
        n_episodes=args.episodes,
        record_video=args.record,
        wandb_log=args.wandb
    )
    print(f"\n[FINAL] Mean Reward: {results['mean']:.2f} +/- {results['std']:.2f}")