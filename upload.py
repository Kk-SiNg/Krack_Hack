from huggingface_hub import HfApi

api = HfApi()
repo_id = "kks25/code-commenter-lora"

# Step 1: Delete all old files
print("Deleting old files...")
files = api.list_repo_files(repo_id)
for f in files:
    if f != ".gitattributes":
        api.delete_file(f, repo_id=repo_id, commit_message=f"Remove old file: {f}")
        print(f"  Deleted: {f}")

# Step 2: Upload new weights
# UPDATE THIS PATH to where you unzipped your new weights
ADAPTER_PATH = r"C:\Users\karti\OneDrive\Desktop\code-commenter"  # <-- CHANGE THIS

api.upload_folder(
    folder_path=ADAPTER_PATH,
    repo_id=repo_id,
    commit_message="Upload new 7B LoRA weights (trained on GFG dataset)"
)

print(f"\n✅ Done! Check: https://huggingface.co/{repo_id}")