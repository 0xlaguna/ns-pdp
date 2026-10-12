"""Write runs/compare.md from the two QLoRA summaries.

Trainable-parameter counts use the day-10 formula (q, k, v, o and gate, up,
down) at Llama 3.2 1B shapes, 8 layers. The log also prints mlx-lm's own
trainable count, which adapts every linear module it finds. Both numbers are
written down. The formula is not a figure read out of the checkpoint.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from count_lora_params import lora_params  # noqa: E402

HIDDEN = 2048
INTERMEDIATE = 8192
LAYERS = 8
TRAINABLE_LINE = re.compile(
    r"Trainable parameters:.*?\(([0-9.]+)M/([0-9.]+)M\)"
)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rank(path: Path) -> int:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return int(data["rank"])


def logged_trainable(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    found = TRAINABLE_LINE.findall(text)
    if not found:
        return None
    trainable, total = found[-1]
    return f"{trainable}M/{total}M"


def fell_faster(summary_a: dict, summary_b: dict, label_a: str, label_b: str) -> str:
    drop_a = summary_a["first_train_loss"] - summary_a["last_train_loss"]
    drop_b = summary_b["first_train_loss"] - summary_b["last_train_loss"]
    if drop_a == drop_b:
        return "Train loss fell by the same amount on both runs."
    winner = label_a if drop_a > drop_b else label_b
    return (
        f"Train loss fell faster on {winner} "
        f"(drop {max(drop_a, drop_b):.3f} vs {min(drop_a, drop_b):.3f})."
    )


def lower_val(summary_a: dict, summary_b: dict) -> tuple[str, str]:
    val_a = summary_a["last_val_loss"]
    val_b = summary_b["last_val_loss"]
    if val_a is None or val_b is None:
        loss_a = summary_a["last_train_loss"]
        loss_b = summary_b["last_train_loss"]
        winner = "A" if loss_a < loss_b else "B" if loss_b < loss_a else "tie"
        sentence = (
            f"Val loss was never printed, so the lower final train loss is run {winner} "
            f"(rank 8 {loss_a}, rank 16 {loss_b})."
        )
        return winner, sentence
    if val_a < val_b:
        winner = "A"
    elif val_b < val_a:
        winner = "B"
    else:
        winner = "tie"
    sentence = (
        f"Run {winner} ended lower on val loss "
        f"(rank 8 {val_a}, rank 16 {val_b})."
    )
    return winner, sentence


def main() -> None:
    summary_a = load_summary(ROOT / "runs" / "run-a.summary.json")
    summary_b = load_summary(ROOT / "runs" / "run-b.summary.json")
    rank_a = load_rank(ROOT / "configs" / "run-a.yaml")
    rank_b = load_rank(ROOT / "configs" / "run-b.yaml")
    params_a = lora_params(HIDDEN, INTERMEDIATE, LAYERS, rank_a)
    params_b = lora_params(HIDDEN, INTERMEDIATE, LAYERS, rank_b)
    logged_a = logged_trainable(ROOT / "runs" / "run-a.log")
    logged_b = logged_trainable(ROOT / "runs" / "run-b.log")
    winner, val_sentence = lower_val(summary_a, summary_b)
    speed_sentence = fell_faster(summary_a, summary_b, "run A", "run B")
    limit = (
        "Two hundred steps on 24 training rows cannot support a general claim "
        "about which rank is better."
    )
    lines = [
        "# Rank 8 vs rank 16",
        "",
        "Both runs use the same data, seed, iters, batch size, learning rate, layer count, and sequence length. Run B adds `--config configs/mlx-run-b.yaml` because `mlx_lm.lora --help` has no `--rank` flag. That file sets `lora_parameters.rank` to 16 and leaves `scale` at the library default 20.0. Rank is the only training change.",
        "",
        "| | run A | run B |",
        "| --- | --- | --- |",
        f"| rank | {rank_a} | {rank_b} |",
        f"| iters | {summary_a['last_step']} | {summary_b['last_step']} |",
        f"| first train loss | {summary_a['first_train_loss']} | {summary_b['first_train_loss']} |",
        f"| last train loss | {summary_a['last_train_loss']} | {summary_b['last_train_loss']} |",
        f"| last val loss | {summary_a['last_val_loss']} | {summary_b['last_val_loss']} |",
        f"| formula trainable params | {params_a} | {params_b} |",
        f"| mlx-lm printed trainable | {logged_a or 'not printed'} | {logged_b or 'not printed'} |",
        "",
        val_sentence,
        speed_sentence,
        limit,
        "",
        f"Lower val loss: run {winner}.",
        "",
    ]
    out = ROOT / "runs" / "compare.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
