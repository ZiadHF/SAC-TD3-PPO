#!/usr/bin/env python3
"""
CMPS458 Assignment 4 - Autonomous RL Training Master
This script runs ALL experiments unattended with crash recovery.
Usage: nohup python run_master.py > master.log 2>&1 &
"""

import os
import sys
import time
import yaml
import torch
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import traceback

# ==================== CONFIGURATION ====================

# YOUR_HF_USERNAME = "ZiadHF" 
# HF_TOKEN = os.getenv("HF_TOKEN")
YOUR_HF_USERNAME = None  # Hugging Face disabled
HF_TOKEN = None  # Hugging Face disabled
WANDB_PROJECT = "cmps458-assignment4_2"

# All experiments to run
EXPERIMENTS = [
    # {"algo": "sac", "env": "LunarLander-v3", "config": "configs/sac_lunarlander.yaml", "priority": 1},
    # {"algo": "td3", "env": "LunarLander-v3", "config": "configs/td3_lunarlander.yaml", "priority": 2},
    # {"algo": "ppo", "env": "LunarLander-v3", "config": "configs/ppo_lunarlander.yaml", "priority": 3},
    {"algo": "sac_cnn", "env": "CarRacing-v3", "config": "configs/sac_carracing.yaml", "priority": 4},
    # {"algo": "td3_cnn", "env": "CarRacing-v3", "config": "configs/td3_carracing.yaml", "priority": 5},
    # {"algo": "ppo_cnn", "env": "CarRacing-v3", "config": "configs/ppo_carracing.yaml", "priority": 6},
]

# ==================== SETUP VALIDATION ====================

def check_environment():
    """Validate system environment before starting"""
    print("Validating environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8+ required")
        return False
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        if gpu_mem < 8:
            print("GPU memory < 8GB - may need to reduce batch_size")
    else:
        print("No GPU detected - training will be very slow")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Check disk space
    free_gb = shutil.disk_usage(".").free // (1024**3)
    print(f"Free disk space: {free_gb}GB")
    if free_gb < 20:
        print("Low disk space - may run out during training")
        return False
    
    # Hugging Face check disabled
    # if not HF_TOKEN:
    #     print("HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")
    #     return False
    
    # Check wandb - either env var or logged in via wandb login
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        try:
            import wandb
            if wandb.api.default_entity:
                print(f"W&B: Logged in as {wandb.api.default_entity}")
            else:
                print("WANDB_API_KEY not set - runs will be anonymous")
        except:
            print("WANDB_API_KEY not set - runs will be anonymous")
    else:
        print("W&B: API key found in environment")
    
    # Check all config files exist
    for exp in EXPERIMENTS:
        if not os.path.exists(exp['config']):
            print(f"Config not found: {exp['config']}")
            return False
    
    print("✅ Environment validation passed\n")
    return True

def setup_huggingface():
    # Hugging Face setup disabled
    print("HF setup skipped (disabled)")
    return

# ==================== EXPERIMENT RUNNER ====================

def run_experiment(exp: Dict[str, Any], log_file) -> bool:
    """Run a single experiment with crash recovery"""
    algo = exp['algo']
    env = exp['env']
    config_path = exp['config']
    
    print(f"\n{'='*70}", file=log_file)
    print(f"🚀 Starting: {algo} on {env}", file=log_file)
    print(f"{'='*70}\n", file=log_file)
    log_file.flush()
    
    # Check if already completed
    completion_marker = f"models/{algo}_{env.replace('-', '_')}_completed.txt"
    if os.path.exists(completion_marker):
        print(f"⚠️  Already completed - skipping {algo}_{env}", file=log_file)
        return True
    
    # Check if interrupted (resume)
    checkpoint_path = f"models/{algo}_{env.replace('-', '_')}_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"📍 Found checkpoint - resuming from {checkpoint_path}", file=log_file)
        resume_args = ["--resume", checkpoint_path]
    else:
        resume_args = []
    
    # Build command
    cmd = [
        sys.executable, "train.py",
        "--config", config_path
    ] + resume_args
    
    # Run with timeout and error handling
    start_time = time.time()
    max_runtime = 12 * 3600  # 12 hour timeout per experiment
    
    try:
        # Use Popen for real-time logging
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "WANDB_MODE": "online"}  # Force GPU 0 and wandb online
        )
        
        # Stream output to log file
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip(), file=log_file)
            log_file.flush()
            
            # Check for errors in output
            if "CUDA out of memory" in line:
                print(f"OOM detected in {algo}_{env} - reducing batch size", file=log_file)
                process.terminate()
                return False
            
            # Check timeout
            if time.time() - start_time > max_runtime:
                print(f"Timeout reached for {algo}_{env}", file=log_file)
                process.terminate()
                return False
        
        process.wait()
        
        if process.returncode == 0:
            # Mark as completed
            with open(completion_marker, 'w') as f:
                f.write(f"Completed at {datetime.now()}\n")
            
            print(f"SUCCESS: {algo}_{env}", file=log_file)
            return True
        else:
            print(f"FAILED: {algo}_{env} (exit code {process.returncode})", file=log_file)
            return False
            
    except Exception as e:
        print(f"CRASH in {algo}_{env}: {e}", file=log_file)
        traceback.print_exc(file=log_file)
        return False

# ==================== MAIN ORCHESTRATOR ====================

class MasterRunner:
    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.master_log = open(self.log_dir / "master.log", "a", encoding='utf-8')  # Add encoding='utf-8'
        
        # Status tracking
        self.status_file = Path("logs/experiment_status.yaml")
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                self.status = yaml.safe_load(f)
        else:
            self.status = {f"{e['algo']}_{e['env']}": "pending" for e in EXPERIMENTS}
    
    def log(self, message):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        print(line, file=self.master_log)
        self.master_log.flush()
    
    def save_status(self):
        """Save current experiment status"""
        with open(self.status_file, 'w') as f:
            yaml.dump(self.status, f)
    
    def run_all(self, resume=False):
        """Run all experiments sequentially"""
        self.log("="*70)
        self.log("MASTER RUNNER STARTED")
        self.log("="*70)
        self.log(f"Total experiments: {len(EXPERIMENTS)}")
        self.log(f"Resume mode: {resume}")
        self.log("="*70 + "\n")
        
        # Sort by priority
        experiments = sorted(EXPERIMENTS, key=lambda x: x['priority'])
        
        for i, exp in enumerate(experiments, 1):
            exp_key = f"{exp['algo']}_{exp['env']}"
            
            # Skip if already completed and not resuming
            if self.status.get(exp_key) == "completed" and not resume:
                self.log(f"Skipping completed: {exp_key}")
                continue
            
            # Skip if failed and not resuming
            if self.status.get(exp_key) == "failed" and not resume:
                self.log(f"Skipping failed: {exp_key}")
                continue
            
            self.log(f"\nExperiment {i}/{len(experiments)}: {exp_key}")
            self.status[exp_key] = "running"
            self.save_status()
            
            # Run experiment
            success = run_experiment(exp, self.master_log)
            
            if success:
                self.status[exp_key] = "completed"
                self.log(f"Marked as completed: {exp_key}")
            else:
                self.status[exp_key] = "failed"
                self.log(f"Marked as failed: {exp_key}")
            
            self.save_status()
            
            # # Cooldown between experiments
            # if i < len(experiments):
            #     self.log("\n💤 Cooling down for 5 minutes...")
            #     time.sleep(300)  # Prevent GPU overheating
        
        # Final summary
        self.print_summary()
        
        # Generate final report
        self.generate_report()
        
        self.log("\nALL EXPERIMENTS FINISHED!")
        self.master_log.close()
    
    def print_summary(self):
        """Print final status summary"""
        self.log("\n" + "="*70)
        self.log("FINAL STATUS SUMMARY")
        self.log("="*70)
        
        completed = sum(1 for s in self.status.values() if s == "completed")
        failed = sum(1 for s in self.status.values() if s == "failed")
        pending = sum(1 for s in self.status.values() if s == "pending")
        
        self.log(f"Completed: {completed}/{len(EXPERIMENTS)}")
        self.log(f"Failed: {failed}/{len(EXPERIMENTS)}")
        self.log(f"Pending: {pending}/{len(EXPERIMENTS)}")
        
        if failed > 0:
            self.log("\nFailed experiments:")
            for exp, status in self.status.items():
                if status == "failed":
                    self.log(f"  - {exp}")
        
        self.log("="*70)
    
    def generate_report(self):
        """Generate a final report from wandb data"""
        self.log("\nGenerating final report...")
        
        try:
            import wandb
            api = wandb.Api(entity="ziadhf-cairo-university")
            
            report_data = []
            for exp in EXPERIMENTS:
                runs = api.runs(
                    WANDB_PROJECT,
                    {"config.algo": exp['algo'], "config.env_id": exp['env']}
                )
                
                if runs:
                    best_run = max(runs, key=lambda r: r.summary.get('eval/mean_reward', -float('inf')))
                    report_data.append({
                        'algorithm': exp['algo'],
                        'environment': exp['env'],
                        'best_score': best_run.summary.get('eval/mean_reward', 'N/A'),
                        'wandb_url': best_run.url
                    })
            
            # Save report
            with open("logs/final_report.md", "w") as f:
                f.write("# CMPS458 Assignment 4 - Final Results\n\n")
                f.write("## Best Scores\n\n")
                for row in report_data:
                    f.write(f"- **{row['algorithm']}** on **{row['environment']}**: "
                           f"{row['best_score']} ([W&B]({row['wandb_url']}))\n")
            
            self.log("Report saved to logs/final_report.md")
            
        except Exception as e:
            self.log(f"Could not generate report: {e}")

# ==================== MAIN ENTRY POINT ====================

def main():
    parser = argparse.ArgumentParser(description="Autonomous RL Master Runner")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--validate-only", action="store_true", help="Only validate setup")
    args = parser.parse_args()
    
    if args.validate_only:
        print("Running validation only...")
        if not check_environment():
            exit(1)
        exit(0)
    
    # Phase 1: Validate environment
    if not check_environment():
        print("\nEnvironment validation failed. Fix issues and retry.")
        exit(1)
    
    # Phase 2: Setup HF (disabled)
    # setup_huggingface()
    print("HF integration is currently disabled.")
    
    # Phase 3: Create directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Phase 4: Run experiments
    runner = MasterRunner()
    runner.run_all(resume=args.resume)
    
    print("\nMaster runner finished! Check logs/master.log for details.")
    print("Final report: logs/final_report.md")

if __name__ == "__main__":
    main()