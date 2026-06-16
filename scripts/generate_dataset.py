import inspect
import json

from mlx_lm import generate, load

import data.raw_functions as funcs


def generate_dataset():
    members = inspect.getmembers(funcs, inspect.isfunction)
    outputs = []
    model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.3")  # type: ignore[reportAssignmentType]
    for name, fn in members:
        source = inspect.getsource(fn)
        messages = [
            {
                "role": "user",
                "content": f"Given the following Python function, write complete unit tests using pytest. Only output the test code, no explanation.\n\n```python\n{source}\n```",
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        text = generate(model, tokenizer, prompt=prompt, verbose=True)

        outputs.append({"prompt": prompt, "completion": text})

    with open("data/output.jsonl", "w+", encoding="utf-8") as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generate_dataset()
