import inspect
import json

from mlx_lm import generate, load

import data.raw_functions as funcs


def generate_dataset():
    members = inspect.getmembers(funcs, inspect.isfunction)
    outputs = []
    model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.3")
    for i, (name, fn) in enumerate(members):
        print(f"[{i + 1}/{len(members)}] generating tests for {name}...")
        source = inspect.getsource(fn)
        messages = [
            {
                "role": "user",
                "content": (
                    "Write pytest unit tests for the following Python function.\n"
                    "Rules:\n"
                    "- Output ONLY valid Python code, nothing else\n"
                    "- No explanations, no markdown, no code fences\n"
                    "- Start directly with import statements\n\n"
                    f"```python\n{source}\n```"
                ),
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # prefix forces the model to continue in code mode
        prompt += "import pytest"

        text = generate(model, tokenizer, prompt=prompt, max_tokens=2048, verbose=False)
        # prepend what we forced so the completion is self-contained
        completion = "import pytest" + text

        outputs.append({"prompt": prompt, "completion": completion})
        print(f"[{i + 1}/{len(members)}] done")

    with open("data/output.jsonl", "w+", encoding="utf-8") as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generate_dataset()
