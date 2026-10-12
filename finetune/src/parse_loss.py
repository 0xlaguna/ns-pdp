"""Pull the first and last train losses, and the last val loss, from an mlx-lm log.

mlx-lm 0.32 prints a rich table, not ``train loss=``. A train row looks like
``10    2.345 ▼`` and a val row like ``1    val 2.100``. Progress redraws and
ANSI codes are stripped before the match.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TRAIN_LINE = re.compile(r"^\s*(\d+)\s+([0-9]+\.[0-9]+)\s+[▼▲]", re.M)
VAL_LINE = re.compile(r"^\s*(\d+)\s+val\s+([0-9]+\.[0-9]+)", re.M)


def clean_log(text: str) -> str:
    text = ANSI.sub("", text)
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.split("\r")[-1]
        lines.append(line)
    return "\n".join(lines)


def parse_losses(text: str) -> dict:
    cleaned = clean_log(text)
    train = [(int(step), float(loss)) for step, loss in TRAIN_LINE.findall(cleaned)]
    val = [(int(step), float(loss)) for step, loss in VAL_LINE.findall(cleaned)]
    if not train:
        raise SystemExit("no train loss lines in log")
    summary = {
        "first_step": train[0][0],
        "last_step": train[-1][0],
        "first_train_loss": train[0][1],
        "last_train_loss": train[-1][1],
        "last_val_loss": val[-1][1] if val else None,
        "last_val_step": val[-1][0] if val else None,
        "train_reports": len(train),
        "val_reports": len(val),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = parse_losses(args.log.read_text(encoding="utf-8", errors="replace"))
    print(json.dumps(summary))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
