"""PEFT LoRA config mirrored from the MLX run.

``--dry-run`` prints rank, alpha, iters, and the trainable-parameter estimate
from the day-10 formula. It does not download a model. ``--train`` is the GPU
path and is not executed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

HIDDEN = 2048
INTERMEDIATE = 8192


def lora_params(hidden: int, intermediate: int, layers: int, rank: int) -> int:
    attn = 4 * rank * (hidden + hidden)
    mlp = 3 * rank * (hidden + intermediate)
    return layers * (attn + mlp)


def config_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "finetune" / "configs" / "run-a.yaml",
        Path("/app/configs/run-a.yaml"),
        Path("configs/run-a.yaml"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("configs/run-a.yaml not found")


def build_config(data: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=int(data["rank"]),
        lora_alpha=int(data["alpha"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    data = yaml.safe_load(config_path().read_text(encoding="utf-8"))
    if args.train and not args.dry_run:
        raise SystemExit("refusing --train on this host; that path is the GPU image")
    if not args.dry_run:
        raise SystemExit("pass --dry-run")
    config = build_config(data)
    params = lora_params(HIDDEN, INTERMEDIATE, int(data["num_layers"]), int(data["rank"]))
    print(f"rank={config.r}")
    print(f"alpha={config.lora_alpha}")
    print(f"iters={int(data['iters'])}")
    print(f"trainable_params={params}")
    print(f"model={data['model']}")


if __name__ == "__main__":
    main()
