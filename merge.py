import json

# Load the old clean data (122 examples)
with open("training_data_clean_old.json", "r", encoding="utf-8") as f:
    old = json.load(f)

# Load the new clean data (from the new run)
with open("training_data_clean.json", "r", encoding="utf-8") as f:
    new = json.load(f)

# Combine
combined = old + new

# Remove duplicates by input
seen = set()
unique = []
for ex in combined:
    if ex["input"] not in seen:
        seen.add(ex["input"])
        unique.append(ex)

with open("training_data_final.json", "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)

print(f"Old: {len(old)} | New: {len(new)} | Combined unique: {len(unique)}")