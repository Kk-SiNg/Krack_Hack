# 🧠 Code Commenter AI

An AI-powered tool that **automatically adds meaningful inline comments** to competitive programming code (C++ and Python).

Built by fine-tuning **Qwen2.5-Coder-7B-Instruct** using **LoRA** on a highly curated, dynamically weighted dataset of algorithmic implementations scraped directly from **GeeksforGeeks**.

**100% local inference — zero API calls to OpenAI, Google, or any external service.**

---

## 📦 Model Weights (Mandatory)

**Google Drive:** [Download Model Weights](https://drive.google.com/drive/folders/10OUfszSL4iYtnNAbYw4Z0VMySEryMO92?usp=sharing)

> Replace the above link with your actual Google Drive sharing link.

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
│                    TRAINING PIPELINE                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. DATA SCRAPING & COLLECTION                          │
│     Web scrape 120+ Data Structure & Algorithm          │
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
│     (DP = 3x weight, Graphs = 2x, Sorting = 1x).        │
│     ↓                                                   │
│  5. FINE-TUNING                                         │
│     Base: Qwen2.5-Coder-7B-Instruct (4-bit quantized)   │
│     Method: LoRA (r=16, alpha=32)                       │
│     Framework: Unsloth + TRL + HuggingFace              │
│     Hardware: Google Colab T4 GPU (free tier)           │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                   INFERENCE PIPELINE                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User pastes code                                       │
│     ↓                                                   │
│  Tokenize with ChatML format (<|im_start|> tags)        │
│     ↓                                                   │
│  Generate with fine-tuned model (temperature=0.2)       │
│     ↓                                                   │
│  Decode and return commented code                       │
│                                                         │
└─────────────────────────────────────────────────────────┘

### Model Details

| Parameter | Value |
|---|---|
| **Base Model** | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| **Quantization** | 4-bit NF4 (bitsandbytes) |
| **Fine-tuning Method** | LoRA (Low-Rank Adaptation) |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Trainable Parameters** | ~40M / 7.6B total (0.53%) |
| **Training Examples** | ~120 curated pairs |
| **Epochs** | 8 |
| **Batch Size** | 1 (gradient accumulation: 8, effective: 8) |
| **Learning Rate** | 2e-4 (cosine schedule) |
| **Optimizer** | AdamW 8-bit |
| **Max Sequence Length** | 768 tokens |
| **Training Time** | ~10 minutes on T4 GPU |
| **Training Framework** | Unsloth + TRL + HuggingFace Transformers |

### Dataset

| Property | Value |
|---|---|
| **Source** | [DeepMind Code Contests](https://huggingface.co/datasets/deepmind/code_contests) |
| **Total Problems** | 13,328 |
| **Languages** | C++ and Python |
| **Filter Criteria** | 5-40 lines, accepted solutions only |
| **Comment Generation** | Qwen2.5-Coder-1.5B via Ollama (100% local) |
| **Cleaning** | Removed wrong comment styles, code rewrites, empty outputs |
| **Final Dataset** | ~120 high-quality input-output pairs |

### Why LoRA?

Full fine-tuning of a 7B parameter model requires multiple A100 GPUs and hundreds of GB of VRAM. **LoRA** (Low-Rank Adaptation) enables fine-tuning by:

1. Freezing all base model weights
2. Injecting small trainable rank-decomposition matrices into attention layers
3. Training only **0.53%** of total parameters
4. Achieving comparable quality to full fine-tuning

This makes it possible to train on a **free Google Colab T4 GPU** (15GB VRAM) in just 10 minutes.

---

## 🚀 How to Run


### Prerequisites

- Python 3.10+
- 16GB RAM (for CPU inference)
- (Optional) NVIDIA GPU with 8GB+ VRAM for faster inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/thestark369/code-commenter.git
cd code-commenter
```

### Step 2: Install Dependencies

**CPU only (no NVIDIA GPU):**
```bash
pip install gradio transformers peft accelerate
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**With NVIDIA GPU:**
```bash
pip install gradio transformers peft accelerate bitsandbytes
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Download Model Weights

Download from **[Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)** and extract:

```bash
# After downloading code-commenter-lora.zip:
unzip code-commenter-lora.zip
```

Your folder structure should look like:
```
code-commenter/
├── code-commenter-lora/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
├── app.py
├── inference.py
└── ...
```

### Step 4: Run the Web App

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

### Step 5: (Alternative) Run from Command Line

```bash
python inference.py
```
## 🌐 Live Demo

**Try it now:** [https://huggingface.co/spaces/kks25/code-commenter](https://huggingface.co/spaces/kks25/code-commenter)

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
code-commenter/
├── app.py                    # Gradio web interface
├── inference.py              # Model loading and prediction
├── generate_local.py         # Dataset generation (Ollama, local)
├── fix_data.py               # Data cleaning and validation
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation (this file)
├── training_data_clean.json  # Cleaned training dataset
├── .gitignore                # Git ignore rules
└── code-commenter-lora/      # Fine-tuned LoRA weights (download separately)
```

| File | Purpose |
|---|---|
| `app.py` | Gradio web app — paste code, get comments |
| `inference.py` | Model loading, tokenization, generation |
| `generate_local.py` | Generates training data using local Ollama |
| `fix_data.py` | Cleans and validates generated training data |
| `training_data_clean.json` | The final cleaned dataset used for training |

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| **Base Model** | Qwen2.5-Coder-7B-Instruct |
| **Fine-tuning** | LoRA + Unsloth + TRL |
| **Quantization** | 4-bit NF4 (bitsandbytes) |
| **Dataset** | DeepMind Code Contests |
| **Data Generation** | Ollama (local LLM inference) |
| **Web UI** | Gradio |
| **Training Hardware** | Google Colab T4 GPU (free) |
| **Languages** | Python |

---

## 📜 License

MIT License

## 👤 Author

**KKS**

Built as an AI/ML project for the Krack Hack Gen AI Challenge.