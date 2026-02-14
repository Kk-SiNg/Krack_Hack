import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CodeCommenter:
    def __init__(self):
        """Load model for CPU inference."""
        # Use the small model for CPU — fast enough for demos
        MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

        print("Loading model... (first time downloads ~3GB, then cached)")
        print("Device: CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
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
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
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