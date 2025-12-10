# Model Saving Guide

## Local Model Saving (Always Works)

Your training code **always saves models locally** regardless of HuggingFace configuration.

### Where Models Are Saved

All models are automatically saved to:
```
models/<algorithm>-<environment>-<timestamp>_<suffix>.pth
```

For example:
```
models/sac_cnn-CarRacing-v3-20251210_141442_best.pth
models/sac_cnn-CarRacing-v3-20251210_141442_latest.pth
models/sac_cnn-CarRacing-v3-20251210_141442_step_100000.pth
```

### Model Types Saved

1. **Best Model** (`*_best.pth`) - Model with highest evaluation score
2. **Latest Model** (`*_latest.pth`) - Most recent checkpoint
3. **Step Checkpoints** (`*_step_N.pth`) - Periodic saves every 5 evaluations

## HuggingFace Uploads (Optional)

HuggingFace uploads are **completely optional** and **disabled by default**.

### Current Configuration

All config files are set to:
```yaml
publish_to_hub: false  # HuggingFace uploads disabled
```

### To Enable HuggingFace Uploads

1. **Set the config**:
   ```yaml
   publish_to_hub: true
   hf_repo_id: "your-username/your-repo-name"
   ```

2. **Set your HuggingFace token**:
   
   **Windows (PowerShell):**
   ```powershell
   $env:HF_TOKEN = "hf_your_token_here"
   ```
   
   **Linux/Mac:**
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

3. **Run training** - models will be uploaded AND saved locally

### What Happens Without HF_TOKEN?

If `publish_to_hub: true` but `HF_TOKEN` is not set:
- ✅ Models still save locally
- ⚠️  Warning message displayed
- ⏭️  Training continues normally

## Running Without HuggingFace

You can run all training completely offline:

```bash
# Just run with default configs (publish_to_hub: false)
python train.py --config configs/sac_carracing.yaml

# Or explicitly disable in config
# Edit config file to set: publish_to_hub: false
```

**Result**: All models saved locally, no HuggingFace dependency needed!

## Loading Saved Models

To load a saved model:

```python
from algorithms import SACAgentCNN
import torch

# Load checkpoint
checkpoint = torch.load('models/sac_cnn-CarRacing-v3-20251210_141442_best.pth')

# Create agent with saved config
agent = SACAgentCNN(**checkpoint['config']['agent_params'])

# Load weights
agent.actor.load_state_dict(checkpoint['actor_state_dict'])
if 'critic_state_dict' in checkpoint:
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])

print(f"Loaded model from step {checkpoint['step']}")
print(f"Best score: {checkpoint['eval_score']:.2f}")
```

## Training Output

You'll see messages like:

```
[START] Training: sac_cnn-CarRacing-v3-20251210_141442
======================================================================
[HF] HuggingFace uploads disabled (publish_to_hub=false)
[SAVE] Models will be saved to: models/sac_cnn-CarRacing-v3-20251210_141442_*.pth

[ENV] Environment: CarRacing-v3
...

[BEST] New best: 156.32
[BEST] Saved model to models/sac_cnn-CarRacing-v3-20251210_141442_best.pth
```

Or with HuggingFace enabled but no token:

```
[HF] ⚠️  publish_to_hub=true but HF_TOKEN not set
[HF] Models will only be saved locally
[HF] To enable uploads: export HF_TOKEN='your_token_here'
...

[BEST] New best: 156.32
[BEST] Saved model to models/sac_cnn-CarRacing-v3-20251210_141442_best.pth
[HF] ⚠️  publish_to_hub=true but HF_TOKEN not set
[HF] Model saved locally at: models/sac_cnn-CarRacing-v3-20251210_141442_best.pth
```

## Summary

✅ **Local saving always works** - no configuration needed  
✅ **HuggingFace is optional** - disabled by default  
✅ **Training never fails** due to HuggingFace issues  
✅ **All models accessible** in the `models/` directory  

You can train completely offline and your models will be safely saved locally!
