import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class CodeCommenter:
    def __init__(self, lora_path: str = None):
        """Load the fine-tuned 7B model with LoRA adapter for inference."""

        BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

        # Resolve LoRA adapter path: local folder > HuggingFace Hub
        if lora_path is None:
            if os.path.isdir("code-commenter-lora"):
                lora_path = "code-commenter-lora"
            else:
                lora_path = "kks25/code-commenter-lora"  # auto-download from HF Hub

        # Auto-detect device
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = torch.float16
            device_label = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            device_map = "cpu"
            dtype = torch.float32
            device_label = "CPU"

        print(f"Loading base model: {BASE_MODEL}")
        print(f"LoRA adapter: {lora_path}")
        print(f"Device: {device_label}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        # Apply LoRA adapter and merge weights for faster inference
        print("Applying LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model ready!\n")

    def comment(self, code: str, max_new_tokens: int = 512) -> str:
        """Add inline comments to the given code."""

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

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        result = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return result.strip()


if __name__ == "__main__":
    import time

    commenter = CodeCommenter()

    test_cpp = """#include <bits/stdc++.h>
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
}"""

    test_py = """n = int(input())
a = list(map(int, input().split()))
a.sort()
print(a[-1] - a[0])"""

    for label, code in [("C++", test_cpp), ("Python", test_py)]:
        print(f"--- {label} Input ---")
        print(code)
        print(f"\n--- {label} Output ---")
        start = time.time()
        result = commenter.comment(code)
        elapsed = time.time() - start
        print(result)
        print(f"\n(Generated in {elapsed:.1f}s)\n")