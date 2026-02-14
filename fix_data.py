import json

# Load checkpoint
with open("checkpoint.json", "r", encoding="utf-8") as f:
    checkpoint = json.load(f)

examples = checkpoint["data"]
print(f"Total examples: {len(examples)}")

clean = []
removed_reasons = {"no_comments": 0, "wrong_comment_style": 0, "code_rewritten": 0, "bad_output": 0}

for ex in examples:
    inp = ex["input"].strip()
    out = ex["output"].strip()
    lang = ex.get("language", "")

    bad = False

    # 1. Output is identical to input (no comments added)
    if inp == out:
        removed_reasons["no_comments"] += 1
        bad = True

    # 2. Python code using // comments instead of #
    if lang == "python" and "//" in out:
        # Count lines with // that aren't in strings
        py_wrong = 0
        for line in out.split("\n"):
            stripped = line.strip()
            if "//" in stripped and not stripped.startswith("#"):
                py_wrong += 1
        if py_wrong > 2:
            removed_reasons["wrong_comment_style"] += 1
            bad = True

    # 3. Output is completely different code (rewritten)
    inp_lines = inp.split("\n")
    out_lines = out.split("\n")
    if abs(len(inp_lines) - len(out_lines)) > 5:
        removed_reasons["code_rewritten"] += 1
        bad = True

    # 4. Contains explanation text
    bad_phrases = ["here's", "here is", "this code", "explanation:",
                   "note:", "```", "the above", "the code",
                   "int main() {", "System.out", "std::"]
    # Only check bad phrases if they shouldn't be there
    if lang == "python":
        for phrase in ["int main", "System.out", "std::", "scanf", "printf", "cout", "cin"]:
            if phrase in out and phrase not in inp:
                removed_reasons["bad_output"] += 1
                bad = True
                break

    # 5. Line count way off
    if len(out_lines) < len(inp_lines) * 0.5:
        removed_reasons["code_rewritten"] += 1
        bad = True

    if not bad:
        clean.append({"input": inp, "output": out})

print(f"\nRemoved:")
for reason, count in removed_reasons.items():
    print(f"  {reason}: {count}")
print(f"\nClean examples: {len(clean)}")

# Save clean data
with open("training_data_clean.json", "w", encoding="utf-8") as f:
    json.dump(clean, f, indent=2, ensure_ascii=False)

print(f"Saved to training_data_clean.json")