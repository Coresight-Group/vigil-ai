import subprocess

print("Pushing to GitHub...")
subprocess.run(["git", "push", "origin", "main"], check=False)

print("\nPushing to Hugging Face Model...")
subprocess.run(["huggingface-cli", "upload", "CoreSightGroup/dual-path-transformer", ".", "--repo-type", "model"], check=False)

print("\nPushing to Hugging Face Space...")
subprocess.run(["huggingface-cli", "upload", "CoreSightGroup/dual-path-transformer", ".", "--repo-type", "space"], check=False)

print("\n" + "="*80)
print("SYNC COMPLETE!")
print("="*80)
print("\nRepositories Updated:")
print("  ✓ GitHub: https://github.com/XE45/HuggingFace")
print("  ✓ HF Model: https://huggingface.co/CoreSightGroup/dual-path-transformer")
print("  ✓ HF Space: https://huggingface.co/spaces/CoreSightGroup/dual-path-transformer")
print("="*80)