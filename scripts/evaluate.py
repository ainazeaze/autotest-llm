import argparse
import json

import mlx.core as mx
from mlx_lm import generate, load


def evaluate(
    model_path: str = "mistralai/Mistral-7B-Instruct-v0.3",
    adapter_path: str = "adapters/",
    data_path: str = "data/valid.jsonl",
    n: int = 3,
    max_tokens: int = 512,
) -> None:
    with open(data_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f][:n]

    prompts = [ex["prompt"] for ex in examples]
    references = [ex["completion"] for ex in examples]

    print("Loading base model...")
    model, tokenizer = load(model_path)
    base_outputs = []
    for i, p in enumerate(prompts):
        print(f"  [{i + 1}/{n}] generating...")
        base_outputs.append(generate(model, tokenizer, prompt=p, max_tokens=max_tokens, verbose=False))
        print(f"  [{i + 1}/{n}] done")

    del model, tokenizer
    mx.metal.clear_cache()

    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load(model_path, adapter_path=adapter_path)
    ft_outputs = []
    for i, p in enumerate(prompts):
        print(f"  [{i + 1}/{n}] generating...")
        ft_outputs.append(generate(ft_model, ft_tokenizer, prompt=p, max_tokens=max_tokens, verbose=False))
        print(f"  [{i + 1}/{n}] done")

    for i, (prompt, base, ft, ref) in enumerate(zip(prompts, base_outputs, ft_outputs, references)):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i + 1}")
        print(f"{'='*60}")
        print(f"\n--- PROMPT ---\n{prompt}\n")
        print(f"--- BASE MODEL ---\n{base}\n")
        print(f"--- FINE-TUNED ---\n{ft}\n")
        print(f"--- REFERENCE ---\n{ref}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter-path", default="adapters/")
    parser.add_argument("--data", default="data/valid.jsonl")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        adapter_path=args.adapter_path,
        data_path=args.data,
        n=args.n,
        max_tokens=args.max_tokens,
    )
