import json
import random
from collections import defaultdict

INPUT_FILE = "data/idiom_sample_ann.json"
OUTPUT_FILE = "data/idiom_eval_human.json"

RANDOM_SEED = 42
SAMPLE_SIZE = 2

random.seed(RANDOM_SEED)

# Load data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Group by polarity + transparency
groups = defaultdict(list)

for item in data:
    key = (item["polarity"], item["transparency"])
    groups[key].append(item)

# Sample 5 items from each group
sampled_data = []

for key in sorted(groups.keys()):
    items = groups[key]

    if len(items) < SAMPLE_SIZE:
        raise ValueError(f"Group {key} has only {len(items)} items, fewer than {SAMPLE_SIZE}.")

    sampled_items = random.sample(items, SAMPLE_SIZE)
    sampled_data.extend(sampled_items)

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(sampled_data)} items to {OUTPUT_FILE}")

# Print group counts for checking
check_counts = defaultdict(int)
for item in sampled_data:
    key = (item["polarity"], item["transparency"])
    check_counts[key] += 1

print("Sampled group counts:")
for key in sorted(check_counts.keys()):
    print(f"{key}: {check_counts[key]}")