# 🧠 Code Commenter AI

An AI-powered tool that **automatically adds meaningful inline comments** to competitive programming code (C++ and Python).

Built by fine-tuning **Qwen2.5-Coder-7B-Instruct** using **LoRA** on a highly curated, dynamically weighted dataset of algorithmic implementations scraped directly from **GeeksforGeeks**.

**100% local inference — zero API calls to OpenAI, Google, or any external service.**

---

## 📦 Model Weights

**HuggingFace Hub (auto-downloaded):** [kks25/code-commenter-lora](https://huggingface.co/kks25/code-commenter-lora)

**Google Drive (manual download):** [Download Model Weights](https://drive.google.com/drive/folders/10OUfszSL4iYtnNAbYw4Z0VMySEryMO92?usp=sharing)

> The inference script automatically downloads LoRA weights from HuggingFace Hub if no local `code-commenter-lora/` folder is found.

---

## 🌐 Live Demo

[open this link to colab deployment](https://a66f929da09d3d4154.gradio.live/)
---

## 🎯 Problem Statement

Competitive programming solutions are notoriously hard to read — they use short variable names, minimal formatting, and zero documentation. This makes them nearly impossible to understand for:
- Students learning algorithms
- Code reviewers
- Anyone revisiting their own old submissions

**Code Commenter AI** solves this by automatically adding concise, meaningful inline comments that explain what each line does — turning cryptic competitive code into readable, documented code.

### Real-World Utility

| Use Case | Impact |
|---|---|
| **Students** | Understand unfamiliar algorithms by reading annotated solutions |
| **Competitive Programmers** | Document solutions for future reference |
| **Educators** | Auto-annotate example code for teaching materials |
| **Code Reviews** | Quickly understand what dense algorithmic code does |

---

## 🔧 How It Works (Technical Overview)

### Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. DATA SCRAPING & COLLECTION                          │
│     Web scrape 122+ Data Structure & Algorithm          │
│     pages from GeeksforGeeks (DP, Graphs, Trees, etc.)  │
│     ↓                                                   │
│  2. GROUND-TRUTH EXTRACTION                             │
│     Parse HTML to find natively commented C++ & Python  │
│     code blocks. Clean raw text & remove HTML junk.     │
│     ↓                                                   │
│  3. INPUT-OUTPUT PAIR GENERATION                        │
│     Use Regex to strip comments from the code.          │
│     Input: Stripped Code → Output: Original GFG Code    │
│     ↓                                                   │
│  4. COMPLEXITY WEIGHTING & BATCHING                     │
│     Oversample hard topics to improve model learning:   │
│     (DP = 3x weight, Graphs = 2x, Sorting = 1x)         │
│     ↓                                                   │
│  5. FINE-TUNING                                         │
│     Base: Qwen2.5-Coder-7B-Instruct (4-bit quantized)   │
│     Method: LoRA (r=16, alpha=32)                       │
│     Framework: Unsloth + TRL + HuggingFace              │
│     Hardware: Google Colab T4 GPU (free tier)           |
│                                                         │
├─────────────────────────────────────────────────────────┤
│                   INFERENCE PIPELINE                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User pastes code                                       │
│     ↓                                                   │
│  Tokenize with ChatML format (<|im_start|> tags)        │
│     ↓                                                   │
│  Load base 7B model (4-bit) + apply LoRA adapter        │
│     ↓                                                   │
│  Generate with fine-tuned model (temperature=0.2)       │
│     ↓                                                   │
│  Decode and return commented code                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Model Details

| Parameter | Value |
|---|---|
| **Base Model** | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| **Quantization** | 4-bit NF4 (bitsandbytes, double quantization) |
| **Fine-tuning Method** | LoRA (Low-Rank Adaptation) |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Trainable Parameters** | 40,370,176 / 7,655,986,688 (0.53%) |
| **Training Examples** | 569 curated pairs from GeeksforGeeks |
| **Epochs** | 5 |
| **Batch Size** | 1 (gradient accumulation: 8, effective: 8) |
| **Total Training Steps** | 360 |
| **Learning Rate** | 2e-4 (cosine schedule) |
| **Optimizer** | AdamW 8-bit |
| **Max Sequence Length** | 768 tokens |
| **Final Training Loss** | 0.019 |
| **Final Validation Loss** | 0.090 |
| **Training Time** | ~72 minutes on T4 GPU |
| **Training Framework** | Unsloth + TRL + HuggingFace Transformers |

### Training Curve

| Step | Training Loss | Validation Loss |
|---|---|---|
| 25 | 0.3517 | 0.3187 |
| 50 | 0.1884 | 0.1914 |
| 75 | 0.1049 | 0.1459 |
| 100 | 0.0935 | 0.1208 |
| 125 | 0.1179 | 0.1004 |
| 150 | 0.0509 | 0.1002 |
| 175 | 0.0508 | 0.0904 |
| 200 | 0.0478 | 0.0843 |
| 225 | 0.0287 | 0.0858 |
| 250 | 0.0249 | 0.0847 |
| 275 | 0.0219 | 0.0842 |
| 300 | 0.0175 | 0.0867 |
| 325 | 0.0187 | 0.0899 |
| 350 | 0.0190 | 0.0901 |

> The model converges well — validation loss plateaus around **0.084** at step ~275. Slight increase after step 300 suggests the best checkpoint is around step 275, though the final model still performs well.

### Dataset

| Property | Value |
|---|---|
| **Source** | 122+ Data Structure & Algorithm pages from GeeksforGeeks |
| **Languages** | C++ and Python |
| **Topics** | Sorting, Searching, Graphs, DP, Trees, Linked Lists, Backtracking, Number Theory, Arrays, Strings, Stacks/Queues, Greedy, Hashing |
| **Filter Criteria** | 5–200 lines, must contain natural inline/block comments |
| **Data Generation** | Input generated by regex-stripping comments from ground-truth GFG code |
| **Cleaning** | Automated HTML parsing, line-number removal, junk text filtering |
| **Complexity Weighting** | Harder topics oversampled (DP = 3×, Graphs = 2×, Sorting = 1×) |
| **Training Split** | 569 examples (95% train / 5% validation) |
| **Final Dataset** | 50-pair batched JSON files in `dataset_batches/` |

### Why LoRA?

Full fine-tuning of a 7B parameter model requires multiple A100 GPUs and hundreds of GB of VRAM. **LoRA** (Low-Rank Adaptation) enables fine-tuning by:

1. Freezing all base model weights
2. Injecting small trainable rank-decomposition matrices into attention layers
3. Training only **0.53%** of total parameters
4. Achieving comparable quality to full fine-tuning

This makes it possible to train on a **free Google Colab T4 GPU** (15GB VRAM) efficiently.

---

## 🚀 How to Run

### Option A: Deploy on Google Colab (Recommended — Free T4 GPU)

This is the easiest way to run the model. No local setup needed.

**Step 1:** Open [Google Colab](https://colab.research.google.com/) and create a new notebook.

**Step 2:** Change runtime to GPU: `Runtime` → `Change runtime type` → **T4 GPU**

**Step 3:** Run the following cells in order:

**Cell 1 — Install dependencies:**
```python
!pip install -q gradio transformers peft accelerate bitsandbytes huggingface_hub
```

**Cell 2 — Clear GPU memory** (run this if you restart or get VRAM errors):
```python
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Free VRAM: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
print(f"Total VRAM: {torch.cuda.mem_get_info()[1] / 1024**3:.1f} GB")
```

**Cell 3 — Load model and launch app:**
Copy the contents of [`colab_cell3_deploy.py`](colab_cell3_deploy.py) and run it.

**Step 4:** Wait ~2-3 minutes for the model to download and load. You'll see:
```
✅ Model ready!
Running on public URL: https://xxxxx.gradio.live
```

**Step 5:** Share the `gradio.live` link — anyone can use it!

> **Note:** The Colab session disconnects after ~90 minutes of inactivity. Re-run the cells to get a new public URL.

> The Colab cell files are available in this repo: [`colab_cell1_install.py`](colab_cell1_install.py), [`colab_cell2_clear_gpu.py`](colab_cell2_clear_gpu.py), [`colab_cell3_deploy.py`](colab_cell3_deploy.py)

---

### Option B: Run Locally

#### Prerequisites

- Python 3.10+
- 16GB+ RAM (for CPU inference)
- (Optional) NVIDIA GPU with 8GB+ VRAM for faster inference

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Kk-SiNg/Krack_Hack.git
cd Krack_Hack
```

#### Step 2: Install Dependencies

**CPU only (no NVIDIA GPU):**
```bash
pip install gradio transformers peft accelerate huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**With NVIDIA GPU:**
```bash
pip install gradio transformers peft accelerate bitsandbytes huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Model Weights

**Automatic (recommended):** Just run the app — LoRA weights are auto-downloaded from [HuggingFace Hub](https://huggingface.co/kks25/code-commenter-lora) on first run.

**Manual download:** Download from [Google Drive](https://drive.google.com/drive/folders/10OUfszSL4iYtnNAbYw4Z0VMySEryMO92?usp=sharing) and extract:

```bash
unzip code-commenter-lora.zip
```

Your folder structure should look like:
```
Krack_Hack/
├── code-commenter-lora/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── app.py
├── inference.py
└── ...
```

#### Step 4: Run the Web App

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

#### Step 5: (Alternative) Run from Command Line

```bash
python inference.py
```

---

## 📸 Demo

### Input (Raw C++ Code)
```cpp
#include <bits/stdc++.h>
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
}
```

### Output (Commented Code)
```cpp
#include <bits/stdc++.h>  // include standard library
using namespace std;  // use standard namespace
int main() {  // main function
    int n;  // declare variable for array size
    cin >> n;  // read array size from input
    vector<int> a(n);  // create vector of size n
    for (int i = 0; i < n; i++)  // loop through each element
        cin >> a[i];  // read each element
    sort(a.begin(), a.end());  // sort array in ascending order
    cout << a[n-1] - a[0] << endl;  // print max minus min
    return 0;  // exit program
}
```

---

## 📁 Project Structure

```
Krack_Hack/
├── app.py                        # Gradio web interface (local)
├── inference.py                  # Model loading (7B + LoRA) and prediction (local)
├── colab_cell1_install.py        # Colab deployment — install dependencies
├── colab_cell2_clear_gpu.py      # Colab deployment — clear GPU memory
├── colab_cell3_deploy.py         # Colab deployment — load model & launch app
├── scrape_gfg.py                 # BeautifulSoup + Cloudscraper — scrapes GFG for training data
├── upload_model.py               # Upload LoRA weights to HuggingFace Hub
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation (this file)
├── Untitled1.ipynb               # Training notebook (Colab)
├── dataset_batches/              # GFG scraped training data in 50-pair batches
├── .gitignore                    # Git ignore rules
└── code-commenter-lora/          # Fine-tuned LoRA weights (auto-downloaded or manual)
```

| File | Purpose |
|---|---|
| `app.py` | Gradio web app for local deployment |
| `inference.py` | Loads Qwen2.5-Coder-7B-Instruct + LoRA adapter, runs generation |
| `colab_cell1_install.py` | Colab Cell 1 — pip install dependencies |
| `colab_cell2_clear_gpu.py` | Colab Cell 2 — clear GPU VRAM |
| `colab_cell3_deploy.py` | Colab Cell 3 — full model loading + Gradio app launch |
| `scrape_gfg.py` | Scrapes naturally commented code from 122+ GFG algorithm pages |
| `upload_model.py` | Uploads LoRA weights to HuggingFace Hub |
| `dataset_batches/` | The final scraped GFG dataset used for training |

> **Note:** Files like `generate_dataset.py`, `generate_local.py`, `merge.py`, `fix_data.py`, `diagnose.py`, and `training_data_clean_old.json` are from earlier experiments and are **not used** in the current pipeline.

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| **Base Model** | Qwen2.5-Coder-7B-Instruct |
| **Fine-tuning** | LoRA + Unsloth + TRL |
| **Quantization** | 4-bit NF4 (bitsandbytes, double quantization) |
| **Dataset** | GeeksforGeeks (122+ algorithm pages, 569 training pairs) |
| **Data Collection** | BeautifulSoup + Cloudscraper |
| **Web UI** | Gradio |
| **Deployment** | Google Colab (T4 GPU) |
| **Training Hardware** | Google Colab T4 GPU (free tier) |
| **Training Time** | ~72 minutes |
| **Languages Supported** | C++ and Python |

---

## 📜 License

MIT License

## 👤 Author

**KKS** ([Kk-SiNg](https://github.com/Kk-SiNg))

Built as an AI/ML project for the Krack Hack Gen AI Challenge.