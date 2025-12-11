from huggingface_hub import create_repo
import os

USERNAME = "USERNAME"  # CHANGE THIS
REPOS = [
    f"{USERNAME}/sac-lunarlander-v3",
    f"{USERNAME}/td3-lunarlander-v3",
    f"{USERNAME}/ppo-lunarlander-v3",
    f"{USERNAME}/sac-carracing-cnn",
    f"{USERNAME}/td3-carracing-cnn",
    f"{USERNAME}/ppo-carracing-cnn"
]

token = os.getenv("HF_TOKEN")
if not token:
    print("❌ Set HF_TOKEN first!")
    exit(1)

for repo in REPOS:
    try:
        create_repo(repo, repo_type="model", exist_ok=True, token=token)
        print(f"✅ {repo}")
    except Exception as e:
        print(f"⚠️  {repo}: {e}")

print("\n🎉 All repos ready!")