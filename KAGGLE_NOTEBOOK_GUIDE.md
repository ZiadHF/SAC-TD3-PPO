# 🎮 Kaggle Notebook Guide

## ✅ Complete! Your notebook is ready for Kaggle

The `kaggle_RL.ipynb` notebook now contains everything you need to run your RL experiments on Kaggle.

## 📦 What's Included

### 1. **All Code Components**
- ✅ Network architectures (MLP, CNN, GaussianPolicy, TwinCritic, etc.)
- ✅ Replay buffers (ReplayBuffer, PPORolloutBuffer)
- ✅ Environment wrappers (PreprocessCarRacing, FrameStack)
- ✅ All agent implementations (SAC, TD3, PPO + CNN variants)
- ✅ Training loop with full functionality
- ✅ Evaluation and visualization utilities

### 2. **Embedded Configurations**
All 6 experiment configs are built-in:
- `sac_carracing` - SAC with CNN on CarRacing
- `td3_carracing` - TD3 with CNN on CarRacing
- `ppo_carracing` - PPO with CNN on CarRacing
- `sac_lunarlander` - SAC with MLP on LunarLander
- `td3_lunarlander` - TD3 with MLP on LunarLander
- `ppo_lunarlander` - PPO with MLP on LunarLander

### 3. **Features**
- 🎯 Easy experiment selection (just change one variable)
- 📊 WandB integration (optional)
- 🎥 Video recording of evaluations
- 💾 Automatic checkpoint saving
- 📈 Progress tracking and logging
- 🔄 Model evaluation utilities

## 🚀 How to Use on Kaggle

### Step 1: Upload
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Click "File" → "Upload Notebook"
4. Select `kaggle_RL.ipynb`

### Step 2: Configure Kaggle Settings
1. **Enable GPU**: 
   - Click "⚙️ Settings" (right sidebar)
   - Accelerator → Select "GPU T4 x2"
   - Click "Save"

2. **Enable Internet** (for WandB - optional):
   - Settings → Internet → Turn ON
   
3. **Add WandB API Key** (optional):
   - Settings → Secrets → Add Secret
   - Name: `WANDB_API_KEY`
   - Value: Your WandB API key from https://wandb.ai/authorize

### Step 3: Select Experiment
In the configuration cell, change:
```python
EXPERIMENT = "sac_carracing"  # Change this!
```

Options:
- `"sac_carracing"` - Best for CarRacing, ~6-8 hours
- `"td3_carracing"` - Alternative for CarRacing, ~6-8 hours
- `"ppo_carracing"` - On-policy variant, ~8-10 hours
- `"sac_lunarlander"` - Quick test, ~1 hour
- `"td3_lunarlander"` - Quick test, ~1 hour
- `"ppo_lunarlander"` - Quick test, ~2-3 hours

### Step 4: Run
Click "Run All" or execute cells sequentially from top to bottom.

## ⚙️ Configuration Options

### Disable WandB
If you don't want to use WandB:
```python
WANDB_ENABLED = False
```

### Disable Video Recording
To save time/space:
```python
ENABLE_VIDEO = False
```

### Change Checkpoint Frequency
```python
CHECKPOINT_FREQ = 50000  # Save every 50K steps instead of 100K
```

## 📊 Monitoring Progress

The notebook will print progress updates every 1000 steps:
```
📊 Step 10,000/2,000,000 (0.5%) | Ep: 15 | R_avg: -45.3 | Best: 120.5
```

Evaluation happens at regular intervals (configured per experiment):
```
🎯 Evaluating at step 25,000...
   Mean: 156.32 ± 45.21
   Range: [89.45, 234.12]
   🌟 New best score!
```

## 💾 Outputs

### Models
Saved in `models/<run_name>/`:
- `best_model.pth` - Model with highest evaluation score
- `checkpoint_100000.pth` - Regular checkpoints
- `checkpoint_200000.pth`
- `final_model.pth` - Final model after training

### Videos
Saved in `videos/<run_name>/`:
- Evaluation episodes recorded as MP4 files
- Can be viewed in the notebook using the video display cells

## 🎓 Post-Training

After training completes, you can:

1. **List all saved models**:
   - Run the "List all saved models" cell

2. **Evaluate a specific model**:
   ```python
   model_path = "models/sac_cnn-CarRacing-v3-20251210_035927/best_model.pth"
   evaluate_model(model_path, n_episodes=10)
   ```

3. **View recorded videos**:
   ```python
   show_videos(run_name="sac_cnn-CarRacing-v3-20251210_035927", max_videos=3)
   ```

4. **Download models**:
   - In Kaggle, click "📁 Output" tab
   - Download the entire `models/` folder
   - Or use Kaggle API to download programmatically

## ⏱️ Estimated Training Times

On Kaggle GPU T4 x2:

| Experiment | Steps | Estimated Time |
|------------|-------|----------------|
| sac_carracing | 2M | 6-8 hours |
| td3_carracing | 2M | 6-8 hours |
| ppo_carracing | 3M | 8-10 hours |
| sac_lunarlander | 300K | 1 hour |
| td3_lunarlander | 300K | 1 hour |
| ppo_lunarlander | 1M | 2-3 hours |

## 🔧 Troubleshooting

### "Out of Memory" error
- Reduce `batch_size` in the config
- Reduce `buffer_size` for SAC/TD3
- Disable video recording

### WandB authentication fails
- Set `WANDB_ENABLED = False` to disable
- Or add API key to Kaggle Secrets

### Training too slow
- Ensure GPU is enabled in settings
- Check device shows "cuda" not "cpu"

### Videos not displaying
- Enable Internet in Kaggle settings
- Or download videos and view locally

## 📝 Notes

- **Kaggle session limit**: Free tier has 12-hour limit, GPU tier has 9-hour limit
- **Checkpoints**: Models are saved every 100K steps, so you won't lose everything if time runs out
- **Resume training**: You can reload the last checkpoint and continue (code supports this)
- **Multiple runs**: Change `EXPERIMENT` and run again for different experiments

## 🎉 Ready to Go!

Your notebook is fully self-contained and ready for Kaggle. Just:
1. Upload
2. Enable GPU
3. Select experiment
4. Run all cells
5. Wait for results

No file uploads, no configuration files, no external dependencies (except packages from pip).

Good luck with your training! 🚀
