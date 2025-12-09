import os
from huggingface_hub import whoami
import torch

# Check HF token
token = os.getenv("HF_TOKEN")
if not token:
    print("❌ HF_TOKEN not set")
    exit(1)

try:
    user = whoami(token=token)
    print(f"✅ HF authenticated: @{user['name']}")
except:
    print("❌ HF token invalid")
    exit(1)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU found, will use CPU")

# Check disk space
import shutil
free_gb = shutil.disk_usage(".").free // (1024**3)
if free_gb < 10:
    print(f"⚠️  Low disk space: {free_gb}GB")
else:
    print(f"✅ Disk space: {free_gb}GB")

print("\n🚀 Ready to train!")