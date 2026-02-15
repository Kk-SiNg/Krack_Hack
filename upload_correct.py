from huggingface_hub import HfApi

api = HfApi()
repo_id = "kks25/code-commenter-lora"

# Step 1: Delete ALL wrong files
print("Deleting old files...")
for f in api.list_repo_files(repo_id):
    if f != ".gitattributes":
        api.delete_file(f, repo_id=repo_id, commit_message=f"Remove: {f}")
        print(f"  Deleted: {f}")

# Step 2: Upload from the CORRECT path
ADAPTER_PATH = r"C:\Users\karti\OneDrive\Desktop\code-commenter\code-commenter-lora"

print(f"\nUploading from: {ADAPTER_PATH}")
api.upload_folder(
    folder_path=ADAPTER_PATH,
    repo_id=repo_id,
    commit_message="Upload new 7B LoRA weights (trained on GFG dataset)"
)

# Step 3: Verify
print("\n✅ Upload complete! Files now in repo:")
for f in api.list_repo_files(repo_id):
    print(f"  {f}")