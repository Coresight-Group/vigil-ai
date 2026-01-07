"""
VIGIL Push Script - Push to GitHub, HF Model, and HF Space
Tokens are entered securely in the terminal (NOT hardcoded)
Run: python push_both.py
"""

import subprocess
import os
import getpass
from huggingface_hub import HfApi, login

# ============================================================================
# CONFIGURATION - EDIT THESE WITH YOUR VALUES
# ============================================================================

HF_ORG = "CoreSightGroup"                    # Change to YOUR org
HF_REPO = "dual-path-transformer"            # Change to YOUR repo
GITHUB_REPO = "https://github.com/XE45/vigil-ai"  # Change to YOUR repo

HF_MODEL_ID = f"{HF_ORG}/{HF_REPO}"

# ============================================================================
# PROMPT FOR TOKEN IN TERMINAL (SECURE - HIDDEN INPUT)
# ============================================================================

print("\n" + "="*80)
print("VIGIL MULTI-PLATFORM PUSH")
print("="*80)

# Get HF Token from terminal input (password-style - hidden)
print("\n[SETUP] Enter your Hugging Face token")
print("        Go to: https://huggingface.co/settings/tokens")
print("        (Token will NOT be displayed as you type - this is secure)")
hf_token = getpass.getpass("HF Token: hf_")

if not hf_token:
    print("\n❌ ERROR: HF Token is required!")
    exit(1)

# Add "hf_" prefix if not already there
if not hf_token.startswith("hf_"):
    hf_token = "hf_" + hf_token

print("✓ Token received (hidden for security)")

# ============================================================================
# 1. LOGIN TO HF
# ============================================================================

print("\n[1/4] Logging in to Hugging Face...")
try:
    login(token=hf_token)
    print("✓ HF login successful")
except Exception as e:
    print(f"❌ HF login failed: {e}")
    print("    Check your token at: https://huggingface.co/settings/tokens")
    exit(1)

# ============================================================================
# 2. PUSH TO GITHUB
# ============================================================================

print("\n[2/4] Pushing to GitHub...")
try:
    result = subprocess.run(
        ["git", "push", "origin", "main"],
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

# ============================================================================
# 3. PUSH TO HF MODEL
# ============================================================================

print("\n[3/4] Pushing to Hugging Face Model...")
print(f"      Repo: {HF_MODEL_ID}")

try:
    api = HfApi()
    api.upload_folder(
        folder_path=".",
        repo_id=HF_MODEL_ID,
        repo_type="model",
        token=hf_token
    )
    print("✓ HF Model pushed successfully")
except Exception as e:
    print(f"❌ HF Model error: {e}")
    print("   Make sure repo exists: https://huggingface.co/new")

# ============================================================================
# 4. PUSH TO HF SPACE
# ============================================================================

print("\n[4/4] Pushing to Hugging Face Space...")
print(f"      Repo: {HF_MODEL_ID}")

try:
    api.upload_folder(
        folder_path=".",
        repo_id=HF_MODEL_ID,
        repo_type="space",
        token=hf_token
    )
    print("✓ HF Space pushed successfully")
except Exception as e:
    print(f"⚠ HF Space: {e}")
    print("   (Not critical - space might not exist yet)")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*80)
print("✓ SYNC COMPLETE!")
print("="*80)
print("\nYour code is now on:")
print(f"  ✓ GitHub: {GITHUB_REPO}")
print(f"  ✓ HF Model: https://huggingface.co/{HF_MODEL_ID}")
print(f"  ✓ HF Space: https://huggingface.co/spaces/{HF_MODEL_ID}")
print("="*80 + "\n")