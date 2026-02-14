import json
import time
import random
import os
import requests

# ============================================================
# CONFIG — NO API KEY NEEDED!
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:1.5b"  # or "qwen2.5-coder:3b" for weaker PCs
TARGET_EXAMPLES = 2000
CHECKPOINT_FILE = "checkpoint.json"
OUTPUT_FILE = "training_data_clean.json"
SYSTEM_PROMPT = (
    "You are a code commenting assistant.\n\n"
    "RULES:\n"
    "1. Return the EXACT same code with inline comments added\n"
    "2. For C++ code: add // comments at the end of lines\n"
    "3. For Python code: add # comments at the end of lines\n"
    "4. NEVER use // in Python. ONLY use # for Python comments\n"
    "5. NEVER rewrite, translate, or change the code\n"
    "6. NEVER add markdown, explanations, or text outside the code\n"
    "7. Keep comments SHORT (under 10 words)\n"
    "8. Every line should get a comment\n\n"
    "CRITICAL: Python uses # for comments. C++ uses // for comments.\n"
    "RETURN ONLY THE COMMENTED CODE."
)

# ============================================================
# STEP 1: Check Ollama is running
# ============================================================
print("=" * 60)
print("Checking Ollama...")
print("=" * 60)

try:
    test = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": False
    }, timeout=30)
    if test.status_code == 200:
        print(f"Ollama is running with {MODEL}!\n")
    else:
        print(f"Error: {test.status_code} {test.text[:200]}")
        print("Make sure Ollama is running: open a terminal and type 'ollama serve'")
        exit(1)
except requests.exceptions.ConnectionError:
    print("Cannot connect to Ollama!")
    print("Fix: Open another terminal and run: ollama serve")
    print("Then re-run this script.")
    exit(1)

# ============================================================
# STEP 2: Load dataset (cached from before!)
# ============================================================
print("=" * 60)
print("STEP 2: Loading dataset (cached)")
print("=" * 60)

from datasets import load_dataset
ds = load_dataset("deepmind/code_contests", split="train", trust_remote_code=True)
print(f"Loaded {len(ds)} problems\n")

# ============================================================
# STEP 3: Extract solutions
# ============================================================
print("=" * 60)
print("STEP 3: Extracting solutions...")
print("=" * 60)

cpp_solutions = []
python_solutions = []
target_per_lang = TARGET_EXAMPLES // 2 + 500

for idx, problem in enumerate(ds):
    solutions = problem.get("solutions", {})
    if not isinstance(solutions, dict):
        continue
    languages = solutions.get("language", [])
    code_list = solutions.get("solution", [])
    if not languages or not code_list:
        continue

    for lang_id, code in zip(languages, code_list):
        if not code or not isinstance(code, str):
            continue
        code = code.strip()
        lines = code.split("\n")
        if len(lines) < 5 or len(lines) > 40:
            continue

        comment_lines = sum(
            1 for l in lines
            if l.strip().startswith("//") or l.strip().startswith("#")
        )
        if comment_lines > len(lines) * 0.3:
            continue

        clean_lines = []
        for line in lines:
            stripped = line.rstrip()
            if "//" in stripped and not stripped.strip().startswith("//"):
                pos = stripped.find("//")
                before = stripped[:pos]
                if before.count('"') % 2 == 0:
                    stripped = before.rstrip()
            clean_lines.append(stripped)
        clean_code = "\n".join(clean_lines)
        if len(clean_code.strip()) < 30:
            continue

        if lang_id == 2 and len(cpp_solutions) < target_per_lang:
            cpp_solutions.append(clean_code)
        elif lang_id in (1, 3) and len(python_solutions) < target_per_lang:
            python_solutions.append(clean_code)

    if len(cpp_solutions) >= target_per_lang and len(python_solutions) >= target_per_lang:
        break
    if (idx + 1) % 500 == 0:
        print(f"  Scanned {idx+1} | C++: {len(cpp_solutions)} | Python: {len(python_solutions)}")

print(f"\nExtracted: {len(cpp_solutions)} C++ | {len(python_solutions)} Python")

all_solutions = []
for c in cpp_solutions:
    all_solutions.append({"code": c, "lang": "cpp"})
for c in python_solutions:
    all_solutions.append({"code": c, "lang": "python"})
random.seed(42)
random.shuffle(all_solutions)
all_solutions = all_solutions[:TARGET_EXAMPLES + 500]
print(f"Total pool: {len(all_solutions)}\n")

# ============================================================
# STEP 4: Comment with LOCAL Ollama — NO LIMITS!
# ============================================================
print("=" * 60)
print(f"STEP 4: Commenting with LOCAL {MODEL}")
print(f"  Target: {TARGET_EXAMPLES}")
print(f"  NO rate limits! Speed depends on your CPU/RAM")
print("=" * 60)


def comment_with_ollama(code_snippet, retries=2):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Add inline comments to this code:\n\n{code_snippet}"}
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 2048
        }
    }

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)

            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(5)
                continue

            data = resp.json()
            output = data["message"]["content"].strip()

            # Clean markdown
            if output.startswith("```"):
                out_lines = output.split("\n")
                out_lines = out_lines[1:]
                if out_lines and out_lines[-1].strip().startswith("```"):
                    out_lines = out_lines[:-1]
                output = "\n".join(out_lines)

            # Validate
            in_count = len(code_snippet.strip().split("\n"))
            out_count = len(output.strip().split("\n"))
            if abs(in_count - out_count) > 3:
                if attempt < retries - 1:
                    continue
                return None

            if len(output.strip()) < 20:
                if attempt < retries - 1:
                    continue
                return None

            return output.strip()

        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(5)

    return None


# Load checkpoint
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    training_data = checkpoint["data"]
    start_idx = checkpoint["next_index"]
    failed = checkpoint.get("failed", 0)
    print(f"\nResumed: {len(training_data)} done, starting at {start_idx}\n")
else:
    training_data = []
    start_idx = 0
    failed = 0

start_time = time.time()

for i in range(start_idx, len(all_solutions)):
    if len(training_data) >= TARGET_EXAMPLES:
        print(f"\nReached target of {TARGET_EXAMPLES}!")
        break

    code = all_solutions[i]["code"]
    lang = all_solutions[i]["lang"]

    t0 = time.time()
    result = comment_with_ollama(code)
    elapsed = time.time() - t0

    if result:
        training_data.append({"input": code, "output": result, "language": lang})
        status = "OK"
    else:
        failed += 1
        status = "FAIL"

    done = len(training_data)
    remaining = TARGET_EXAMPLES - done
    avg_time = (time.time() - start_time) / max(1, done)
    eta = int(remaining * avg_time / 60)

    if (i + 1) % 5 == 0:
        print(
            f"  [{i+1}] {status} | "
            f"Done: {done}/{TARGET_EXAMPLES} | "
            f"Failed: {failed} | "
            f"Speed: {elapsed:.1f}s/example | "
            f"ETA: ~{eta}min"
        )

    if (i + 1) % 25 == 0:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump({"data": training_data, "next_index": i + 1, "failed": failed}, f, ensure_ascii=False)
        print(f"  >> Checkpoint saved ({done} examples)")

# ============================================================
# STEP 5: Validate
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Validating...")
print("=" * 60)

clean_data = []
removed = 0
for item in training_data:
    inp = item["input"].strip()
    out = item["output"].strip()
    bad = False
    if abs(len(inp.split("\n")) - len(out.split("\n"))) > 2:
        bad = True
    bad_phrases = ["here's", "here is", "this code", "explanation:", "note:", "```", "the above"]
    if any(p in out.lower() for p in bad_phrases):
        bad = True
    comment_lens = []
    for line in out.split("\n"):
        if "//" in line:
            comment_lens.append(len(line[line.rfind("//"):]))
        elif "#" in line and not line.strip().startswith("#"):
            comment_lens.append(len(line[line.rfind("#"):]))
    if comment_lens and sum(comment_lens) / len(comment_lens) > 60:
        bad = True
    if bad:
        removed += 1
    else:
        clean_data.append({"input": inp, "output": out})

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, indent=2, ensure_ascii=False)
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

total_time = int((time.time() - start_time) / 60)
print(f"\n  Total time: {total_time} minutes")
print(f"  Generated: {len(training_data)}")
print(f"  Removed: {removed}")
print(f"  Clean saved: {len(clean_data)}")
print(f"  File: {OUTPUT_FILE}")
print(f"\n  Upload {OUTPUT_FILE} to Colab for training!")