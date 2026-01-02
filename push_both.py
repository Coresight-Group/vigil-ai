import subprocess
import os
from huggingface_hub import HfApi, login

# Get token from environment (NOT hardcoded!)
hf_token = os.environ.get('HF_TOKEN')

if not hf_token:
    print("ERROR: Set HF_TOKEN first!")
    exit(1)

print("Logging in to Hugging Face...")
login(token=hf_token)

print("\nPushing to GitHub...")
subprocess.run(["git", "push", "origin", "main"], check=False)

print("\nPushing to HF Model...")
api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id="CoreSightGroup/dual-path-transformer",
    repo_type="model",
    token=hf_token
)
print("✓ Model pushed successfully")

print("\nPushing to HF Space...")
api.upload_folder(
    folder_path=".",
    repo_id="CoreSightGroup/dual-path-transformer",
    repo_type="space",
    token=hf_token
)
print("✓ Space pushed successfully")

print("\n" + "="*80)
print("SYNC COMPLETE!")
print("="*80)
print("\nRepositories Updated:")
print("  ✓ GitHub: https://github.com/XE45/HuggingFace")
print("  ✓ HF Model: https://huggingface.co/CoreSightGroup/dual-path-transformer")
print("  ✓ HF Space: https://huggingface.co/spaces/CoreSightGroup/dual-path-transformer")
print("="*80)