# 🎮 Quick Reference Card

## 🎯 Change Experiment (Line ~15 in config cell)
```python
EXPERIMENT = "sac_carracing"  # ← CHANGE THIS
```

## 📊 Available Experiments

| Name | Environment | Algorithm | Steps | Time (GPU) |
|------|-------------|-----------|-------|------------|
| `sac_carracing` | CarRacing-v3 | SAC+CNN | 2M | 6-8h |
| `td3_carracing` | CarRacing-v3 | TD3+CNN | 2M | 6-8h |
| `ppo_carracing` | CarRacing-v3 | PPO+CNN | 3M | 8-10h |
| `sac_lunarlander` | LunarLander-v3 | SAC+MLP | 300K | 1h |
| `td3_lunarlander` | LunarLander-v3 | TD3+MLP | 300K | 1h |
| `ppo_lunarlander` | LunarLander-v3 | PPO+MLP | 1M | 2-3h |

## ⚙️ Quick Settings

### Disable WandB (if you don't want logging)
```python
WANDB_ENABLED = False
```

### Disable Videos (to save space/time)
```python
ENABLE_VIDEO = False
```

### Change Your WandB Username
```python
WANDB_ENTITY = "your-username-here"
```

## 🚀 Kaggle Setup Checklist

- [ ] Upload notebook to Kaggle
- [ ] Enable GPU (Settings → Accelerator → GPU T4 x2)
- [ ] Enable Internet (Settings → Internet → ON) - *Optional, for WandB*
- [ ] Add WandB API Key to Secrets - *Optional*
- [ ] Change `EXPERIMENT` variable
- [ ] Click "Run All"

## 📦 Outputs

**Models**: `models/<run_name>/`
- `best_model.pth` - Best performing model
- `checkpoint_*.pth` - Regular saves every 100K steps
- `final_model.pth` - Final model

**Videos**: `videos/<run_name>/`
- Evaluation episodes as MP4 files

## 💡 Pro Tips

1. **Start with LunarLander** - Quick (1 hour) to verify everything works
2. **Use GPU** - 5-10x faster than CPU
3. **Enable checkpoints** - Don't lose progress if Kaggle times out
4. **Monitor WandB** - Track training curves in real-time
5. **Download models** - Use Kaggle Output tab or API

## 🔍 Key Hyperparameters

### SAC CarRacing (Best Settings)
- Learning rate: 0.0001
- Batch size: 256
- Buffer size: 200K
- Learning starts: 35K

### TD3 CarRacing
- Learning rate: 0.0001
- Batch size: 256
- Exploration noise: 0.3
- Learning starts: 35K

### PPO CarRacing
- Learning rate: 0.0003
- Batch size: 128
- Rollout length: 4096
- Train iterations: 15

## 🎯 Expected Performance

### CarRacing-v3
- **Solving criterion**: Average reward > 900
- **SAC**: Usually solves in 1-1.5M steps
- **TD3**: Usually solves in 1-1.5M steps
- **PPO**: Usually solves in 2-2.5M steps

### LunarLander-v3
- **Solving criterion**: Average reward > 200
- **SAC**: Usually solves in 100-200K steps
- **TD3**: Usually solves in 100-200K steps
- **PPO**: Usually solves in 400-600K steps

## 🆘 Common Issues

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size` or `buffer_size` |
| Training too slow | Enable GPU in Kaggle settings |
| WandB fails | Set `WANDB_ENABLED = False` |
| Videos don't play | Enable Internet in settings |
| Session timeout | Models auto-saved, reload checkpoint |

## 📱 Check Training Progress

Look for these console outputs:
```
📊 Step 10,000/2,000,000 (0.5%) | Ep: 15 | R_avg: -45.3 | Best: 120.5
```

And evaluation results:
```
🎯 Evaluating at step 25,000...
   Mean: 156.32 ± 45.21
   🌟 New best score!
```

## ✅ All Done Checklist

After training:
- [ ] Check final best score
- [ ] Download models from Output tab
- [ ] Download videos (optional)
- [ ] Check WandB dashboard for training curves
- [ ] Evaluate model on test episodes
- [ ] Share results! 🎉

---

**Need help?** Check `KAGGLE_NOTEBOOK_GUIDE.md` for detailed instructions.
