import os
from huggingface_hub import upload_file, create_repo
from huggingface_hub.utils import validate_repo_id, HfHubHTTPError
import time
import hashlib

class RobustHFUploader:
    def __init__(self, token=None):
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("❌ HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")
    
    def upload_with_retry(self, local_path, repo_id, path_in_repo="model.pth", max_retries=5):
        if not os.path.exists(local_path):
            print(f"❌ File not found: {local_path}")
            return False
        
        try:
            validate_repo_id(repo_id)
        except:
            print(f"❌ Invalid repo ID: {repo_id}")
            return False
        
        # Create repo if it doesn't exist
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True, token=self.token)
        except:
            pass
        
        # Calculate checksum
        with open(local_path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                print(f"📤 Uploading... (attempt {attempt+1}/{max_retries})")
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="model",
                    token=self.token,
                    commit_message=f"Model upload (MD5: {checksum})"
                )
                print("✅ Upload successful!")
                return True
            except HfHubHTTPError as e:
                if e.response.status_code == 429:
                    wait = 60 * (2 ** attempt)
                    print(f"⏳ Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"❌ HTTP Error: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(30)
        
        print("❌ Max retries exceeded")
        return False

# Global instance
uploader = RobustHFUploader()