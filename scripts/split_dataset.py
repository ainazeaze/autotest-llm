import json
import random

TRAIN_PATH = "train.jsonl"
VALID_PATH = "valid.jsonl"


def train_valid_split(file_path: str) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    shuffled = data.copy()
    random.seed(42)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.8)

    train = shuffled[:split_idx]
    valid = shuffled[split_idx:]

    with open("data/train.jsonl", "w+", encoding="utf-8") as f:
        for line in train:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    with open("data/valid.jsonl", "w+", encoding="utf-8") as f:
        for line in valid:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    train_valid_split("data/output.jsonl")
