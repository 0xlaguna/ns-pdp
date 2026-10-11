"""Trainable LoRA parameter count for the layers this run adapts.

Each adapted linear of shape (d_in, d_out) contributes rank * (d_in + d_out).
Attention uses q, k, v, and o, all hidden by hidden. The MLP uses gate, up,
and down, each hidden by intermediate. The count is linear in rank, so rank
16 is twice rank 8.
"""

from __future__ import annotations

import argparse


def lora_params(hidden: int, intermediate: int, layers: int, rank: int) -> int:
    attn = 4 * rank * (hidden + hidden)
    mlp = 3 * rank * (hidden + intermediate)
    return layers * (attn + mlp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--reference-params", type=int, required=True)
    args = parser.parse_args()

    other_rank = args.rank * 2
    for rank in (args.rank, other_rank):
        params = lora_params(args.hidden, args.intermediate, args.layers, rank)
        fraction = params / args.reference_params
        print(f"rank {rank} trainable params: {params}")
        print(f"rank {rank} fraction: {fraction}")


if __name__ == "__main__":
    main()
