"""
VIGIL Push Script - Push to GitHub
Run: python push_both.py
"""

import subprocess

GITHUB_REPO = "https://github.com/Coresight-Group/vigil-ai"

print("\n" + "="*60)
print("VIGIL GITHUB PUSH")
print("="*60)

print("\nPushing to GitHub...")
try:
    result = subprocess.run(
        ["git", "push", "origin", "master:main"],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode == 0:
        print("✓ GitHub pushed successfully")
    else:
        print(f"⚠ GitHub: {result.stderr.strip()}")
except FileNotFoundError:
    print("⚠ Git not found - did you run 'git init' first?")
except Exception as e:
    print(f"⚠ GitHub error: {e}")

print("\n" + "="*60)
print("✓ DONE!")
print(f"  GitHub: {GITHUB_REPO}")
print("="*60 + "\n")
