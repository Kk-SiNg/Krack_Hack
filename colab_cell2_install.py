# Cell 2: Clear GPU memory (run after restart or if you get VRAM errors)
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Free VRAM: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
print(f"Total VRAM: {torch.cuda.mem_get_info()[1] / 1024**3:.1f} GB")