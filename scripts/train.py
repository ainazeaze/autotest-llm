import argparse
import subprocess


def train(
    model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    data: str = "data/",
    iters: int = 100,
    batch_size: int = 2,
    num_layers: int = 8,
    learning_rate: float = 1e-4,
    val_batches: int = 5,
    steps_per_eval: int = 25,
    adapter_path: str = "adapters/",
) -> None:
    subprocess.run(
        [
            "python", "-m", "mlx_lm", "lora",
            "--model", model,
            "--train",
            "--data", data,
            "--iters", str(iters),
            "--batch-size", str(batch_size),
            "--num-layers", str(num_layers),
            "--learning-rate", str(learning_rate),
            "--val-batches", str(val_batches),
            "--steps-per-eval", str(steps_per_eval),
            "--adapter-path", adapter_path,
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--data", default="data/")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-batches", type=int, default=5)
    parser.add_argument("--steps-per-eval", type=int, default=25)
    parser.add_argument("--adapter-path", default="adapters/")
    args = parser.parse_args()

    train(
        model=args.model,
        data=args.data,
        iters=args.iters,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        val_batches=args.val_batches,
        steps_per_eval=args.steps_per_eval,
        adapter_path=args.adapter_path,
    )
