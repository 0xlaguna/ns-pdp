# Rank 8 vs rank 16

Both runs use the same data, seed, iters, batch size, learning rate, layer count, and sequence length. Run B adds `--config configs/mlx-run-b.yaml` because `mlx_lm.lora --help` has no `--rank` flag. That file sets `lora_parameters.rank` to 16 and leaves `scale` at the library default 20.0. Rank is the only training change.

| | run A | run B |
| --- | --- | --- |
| rank | 8 | 16 |
| iters | 200 | 200 |
| first train loss | 3.373 | 3.406 |
| last train loss | 0.038 | 0.091 |
| last val loss | 3.507 | 3.608 |
| formula trainable params | 3014656 | 6029312 |
| mlx-lm printed trainable | 2.818M/1235.814M | 5.636M/1235.814M |

Run A ended lower on val loss (rank 8 3.507, rank 16 3.608).
Train loss fell faster on run A (drop 3.335 vs 3.315).
Two hundred steps on 24 training rows cannot support a general claim about which rank is better.

Lower val loss: run A.
