import json
import time
import random
import os
import requests


# ============================================================
# CONFIG
# ============================================================
GROQ_API_KEY = "enter your api key"  # <-- paste your Groq key here
TARGET_EXAMPLES = 2000
DELAY = 4  # 2.5seconds = ~24 req/min (safely under 30/min limit)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"  # powerful, free on Groq

CHECKPOINT_FILE = "checkpoint.json"
OUTPUT_FILE = "training_data_clean.json"

SYSTEM_PROMPT = (
    "You are a MINIMAL code commenting assistant for competitive programmers.\n\n"
    "STRICT RULES:\n"
    "1. Return the EXACT same code. Do NOT change, reformat, add, or remove ANY code\n"
    "2. Add a SHORT inline comment (5-10 words MAX) at the END of each line\n"
    "3. Use // for C++ and # for Python\n"
    "4. Comments explain PURPOSE or KEY INSIGHT briefly\n"
    "5. Do NOT add any text, explanation, or markdown before or after the code\n"
    "6. Do NOT add comments on separate lines. ONLY at the end of existing lines\n"
    "7. Do NOT wrap output in code blocks\n"
    "8. Keep EXACT original formatting\n"
    "9. Every single line must get a comment\n\n"
    "RETURN ONLY THE COMMENTED CODE. NOTHING ELSE."
)

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}


# ============================================================
# STEP 1: Load dataset (uses cache from your previous run!)
# ============================================================
print("=" * 60)
print("STEP 1: Loading dataset (cached from previous download)")
print("=" * 60)

from datasets import load_dataset
ds = load_dataset("deepmind/code_contests", split="train", trust_remote_code=True)
print(f"Loaded {len(ds)} problems\n")


# ============================================================
# STEP 2: Extract solutions
# ============================================================
print("=" * 60)
print("STEP 2: Extracting clean solutions...")
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

        # Strip existing inline comments
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
# STEP 3: Comment with Groq API
# ============================================================
print("=" * 60)
print("STEP 3: Commenting with Groq (Llama 3.3 70B)...")
print(f"  Target: {TARGET_EXAMPLES} examples")
print(f"  Delay: {DELAY}s between calls")
est = int(TARGET_EXAMPLES * DELAY / 60)
print(f"  Estimated time: ~{est} minutes")
print("=" * 60)

def comment_with_groq(code_snippet, retries=3):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Add inline comments to this code:\n\n{code_snippet}"}
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=60)

            if resp.status_code == 429:
                # Parse the reset time from headers
                reset = resp.headers.get("x-ratelimit-reset-requests", "")
                # Extract minutes from format like "4h0m28.8s"
                wait_seconds = 60  # default
                if "h" in reset:
                    # Hit daily/hourly limit — wait for full reset
                    print(f"\n    QUOTA HIT! Resets in: {reset}")
                    print(f"    Auto-sleeping for 10 minutes then retrying...")
                    print(f"    (Script will keep retrying automatically. Leave it running!)\n")
                    time.sleep(600)  # wait 10 min, then retry
                    continue
                elif "m" in reset:
                    # Hit per-minute limit — short wait
                    wait_seconds = 65
                    print(f"    Per-minute limit. Waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue
                else:
                    time.sleep(30)
                    continue

            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(10)
                continue

            data = resp.json()
            output = data["choices"][0]["message"]["content"].strip()

            # Clean markdown wrappers
            if output.startswith("```"):
                out_lines = output.split("\n")
                out_lines = out_lines[1:]
                if out_lines and out_lines[-1].strip().startswith("```"):
                    out_lines = out_lines[:-1]
                output = "\n".join(out_lines)

            # Validate line count
            in_count = len(code_snippet.strip().split("\n"))
            out_count = len(output.strip().split("\n"))
            if abs(in_count - out_count) > 3:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return None

            if len(output.strip()) < 20:
                continue

            return output.strip()

        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"    Error: {e} | Retrying in {wait}s...")
            time.sleep(wait)

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


for i in range(start_idx, len(all_solutions)):
    if len(training_data) >= TARGET_EXAMPLES:
        print(f"\nReached target of {TARGET_EXAMPLES}!")
        break

    code = all_solutions[i]["code"]
    lang = all_solutions[i]["lang"]

    result = comment_with_groq(code)

    if result:
        training_data.append({
            "input": code,
            "output": result,
            "language": lang
        })
        status = "OK"
    else:
        failed += 1
        status = "FAIL"

    done = len(training_data)
    remaining = TARGET_EXAMPLES - done
    eta = int(remaining * DELAY / 60)

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}] {status} | Done: {done}/{TARGET_EXAMPLES} | Failed: {failed} | ETA: ~{eta}min")

    if (i + 1) % 50 == 0:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump({"data": training_data, "next_index": i + 1, "failed": failed}, f, ensure_ascii=False)
        print(f"  >> Checkpoint saved ({done} examples)")

    time.sleep(DELAY)


# ============================================================
# STEP 4: Validate and save
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Validating...")
print("=" * 60)

clean_data = []
removed = 0

for item in training_data:
    inp = item["input"].strip()
    out = item["output"].strip()

    bad = False

    if abs(len(inp.split("\n")) - len(out.split("\n"))) > 2:
        bad = True

    bad_phrases = ["here's", "here is", "this code", "explanation:",
                   "note:", "```", "the above", "the code"]
    if any(p in out.lower() for p in bad_phrases):
        bad = True

    comment_lens = []
    for line in out.split("\n"):
        if "//" in line:
            comment_lens.append(len(line[line.rfind("//"):]))
        elif "#" in line and not line.strip().startswith("#"):
            comment_lens.append(len(line[line.rfind("#"):]))
    if comment_lens and sum(comment_lens)/len(comment_lens) > 60:
        bad = True

    if bad:
        removed += 1
    else:
        clean_data.append({"input": inp, "output": out})

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, indent=2, ensure_ascii=False)

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"\n  Generated: {len(training_data)}")
print(f"  Removed (bad): {removed}")
print(f"  Clean saved: {len(clean_data)}")
print(f"  File: {OUTPUT_FILE}")
print(f"\n  Upload {OUTPUT_FILE} to Colab for training!")