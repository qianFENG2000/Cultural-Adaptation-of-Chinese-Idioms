import json
import random
from pathlib import Path


INPUT_PATH = Path("data/idiom_base.json")
OUTPUT_PATH = Path("data/idiom_sample.json")
SAMPLE_SIZE = 100
RANDOM_SEED = 42


def main() -> None:
    random.seed(RANDOM_SEED)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON array.")

    if len(data) < SAMPLE_SIZE:
        raise ValueError(
            f"Not enough samples in input file: found {len(data)}, "
            f"but need at least {SAMPLE_SIZE}."
        )

    sampled = random.sample(data, SAMPLE_SIZE)

    sampled_output = []
    for item in sampled:
        sampled_output.append(
            {
                "sentence": item["sentence"],
                "index": item["index"],
                "source_idiom": item["source_idiom"],
            }
        )

    sampled_output = sorted(sampled_output, key=lambda x: x["index"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sampled_output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(sampled_output)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()