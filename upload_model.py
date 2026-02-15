from huggingface_hub import HfApi

api = HfApi()

# Change this to YOUR HuggingFace username
HF_USERNAME = "kks25"

# Create model repo
repo_id = f"{HF_USERNAME}/code-commenter-lora"
api.create_repo(repo_id, exist_ok=True)

# Upload the NEW 7B LoRA weights (trained on GFG + DeepMind combined dataset)
api.upload_folder(
    folder_path="code-commenter-lora",
    repo_id=repo_id,
    commit_message="Upload fine-tuned 7B LoRA weights (new dataset)"
)

print(f"\n✅ Model uploaded to: https://huggingface.co/{repo_id}")