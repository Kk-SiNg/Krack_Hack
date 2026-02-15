import gradio as gr
from inference import CodeCommenter
import time

# ============================================================
# Load model once at startup
# ============================================================
print("Starting Code Commenter...")
commenter = CodeCommenter()

# ============================================================
# Examples
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
    ["""#include <bits/stdc++.h>
using namespace std;
int main() {
    int t;
    cin >> t;
    while (t--) {
        int l, r, d;
        cin >> l >> r >> d;
        if (d < l)
            cout << d << endl;
        else
            cout << ((r / d) + 1) * d << endl;
    }
}"""],
]


def add_comments(code: str) -> str:
    if not code or not code.strip():
        return "Please paste some code first!"

    start = time.time()
    result = commenter.comment(code)
    elapsed = time.time() - start

    return result + f"\n\n# --- Generated in {elapsed:.1f}s ---"


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(
    title="Code Commenter AI",
) as app:

    gr.Markdown("""
    # 🧠 Code Commenter AI

    **Paste your competitive programming code** and get instant inline comments.

    Fine-tuned on naturally commented code scraped from
    **122+ GeeksforGeeks** algorithm pages. Supports **C++** and **Python**.

    *Running on CPU — expect ~30-60 seconds per prediction. Much faster on GPU.*
    """)

    with gr.Row():
        with gr.Column():
            input_code = gr.Code(
                label="📥 Paste your code here",
                language=None,
                lines=20,
            )
            btn = gr.Button(
                "✨ Add Comments",
                variant="primary",
                size="lg"
            )

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

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )