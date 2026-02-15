from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download("kks25/code-commenter-lora", "adapter_config.json")
with open(config_path) as f:
    config = json.load(f)

print(f"Base model: {config.get('base_model_name_or_path', 'NOT FOUND')}")
print(f"LoRA rank:  {config.get('r', 'NOT FOUND')}")
print(f"LoRA alpha: {config.get('lora_alpha', 'NOT FOUND')}")