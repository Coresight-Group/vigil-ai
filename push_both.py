from huggingface_hub import login, upload_folder
import subprocess

# Push to Hugging Face
login()
upload_folder(folder_path=".", repo_id="XE45/UnstructuredData", repo_type="model")

# Push to GitHub
subprocess.run(["git", "push", "github", "main"])