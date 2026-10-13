# Fine-tune

Rank-8 QLoRA command, from `runs/run-a.cmd`:

```bash
.venv/bin/mlx_lm.lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train \
  --data data \
  --adapter-path adapters/run-a \
  --iters 200 \
  --batch-size 1 \
  --num-layers 8 \
  --max-seq-length 512 \
  --seed 7 \
  --learning-rate 1e-4
```

Run B is the same command with `--adapter-path adapters/run-b` and `--config configs/mlx-run-b.yaml`.
