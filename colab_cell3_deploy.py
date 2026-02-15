# Cell 3: Load model and launch Gradio app
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import snapshot_download
import time

# ============================================================
# Config
# ============================================================
LORA_REPO = "kks25/code-commenter-lora"
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# ============================================================
# Load model in 4-bit (same as training)
# ============================================================
print(f"Loading base model: {BASE_MODEL}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Free VRAM: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)

# Load LoRA weights
print(f"Downloading LoRA weights from {LORA_REPO}...")
lora_path = snapshot_download(repo_id=LORA_REPO)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print("✅ Model ready!\n")

# ============================================================
# Inference
# ============================================================
def add_comments(code: str) -> str:
    if not code or not code.strip():
        return "Please paste some code first!"

    prompt = (
        "<|im_start|>system\n"
        "You are a code commenting assistant. Add short inline comments "
        "to the end of each line of code. "
        "Use // for C++ and # for Python. Keep comments under 10 words. "
        "Return ONLY the commented code.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Add inline comments to this code:\n\n{code}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    elapsed = time.time() - start
    return result + f"\n\n# --- Generated in {elapsed:.1f}s ---"

# ============================================================
# Gradio UI
# ============================================================
EXAMPLES = [
    ["""#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++)
        cin >> a[i];
    sort(a.begin(), a.end());
    cout << a[n-1] - a[0] << endl;
    return 0;
}"""],
    ["""n = int(input())
a = list(map(int, input().split()))
a.sort()
print(a[-1] - a[0])"""],
    ["""def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

t = int(input())
for _ in range(t):
    a, b = map(int, input().split())
    print(gcd(a, b))"""],
]

with gr.Blocks(title="Code Commenter AI") as app:

    gr.Markdown("""
    # 🧠 Code Commenter AI

    **Paste your competitive programming code** and get instant inline comments.

    Fine-tuned **Qwen2.5-Coder-7B-Instruct** with LoRA on naturally commented code
    scraped from **122+ GeeksforGeeks** algorithm pages.

    Supports **C++** and **Python**. Running on **Colab T4 GPU** — expect ~3-8s per prediction.
    """)

    with gr.Row():
        with gr.Column():
            input_code = gr.Code(
                label="📥 Paste your code here",
                language=None,
                lines=20,
            )
            btn = gr.Button("✨ Add Comments", variant="primary", size="lg")

        with gr.Column():
            output_code = gr.Code(
                label="📤 Commented code",
                language=None,
                lines=20,
            )

    btn.click(fn=add_comments, inputs=input_code, outputs=output_code)

    gr.Markdown("### 📝 Try these examples:")
    gr.Examples(examples=EXAMPLES, inputs=input_code)

    gr.Markdown("""
    ---
    **Tech Stack:** Qwen2.5-Coder-7B-Instruct | LoRA Fine-tuning | Unsloth | Gradio

    Built by **Kk-SiNg** |
    [GitHub](https://github.com/Kk-SiNg/Krack_Hack)
    """)

app.launch(share=True)