# Reciprocal rank fusion

RRF constant k=60. Each parent list is that index's top 20.
A fused top-5 differs from both parents when its id order matches neither parent top-5.

## What is a cross-encoder?

- dense top-5: retrieval-notes:2, retrieval-notes:5, retrieval-notes:0, retrieval-notes:1, retrieval-notes:3
- lexical top-5: retrieval-notes:5, retrieval-notes:2, fieldguide-runbook:1, mlx-lora-notes:4, retrieval-notes:3
- fused top-5: retrieval-notes:2, retrieval-notes:5, retrieval-notes:0, retrieval-notes:1, retrieval-notes:3
- differs from both parents: false

| fused id | score | dense_rank | lexical_rank |
| --- | --- | --- | --- |
| retrieval-notes:2 | 0.032522 | 1 | 2 |
| retrieval-notes:5 | 0.032522 | 2 | 1 |
| retrieval-notes:0 | 0.030798 | 3 | 7 |
| retrieval-notes:1 | 0.030777 | 4 | 6 |
| retrieval-notes:3 | 0.030769 | 5 | 5 |

## mlx_lm.lora

- dense top-5: mlx-lora-notes:0, mlx-lora-notes:1, mlx-lora-notes:5, mlx-lora-notes:2, retrieval-notes:5
- lexical top-5: mlx-lora-notes:0, mlx-lora-notes:5, mlx-lora-notes:1, mlx-lora-notes:2
- fused top-5: mlx-lora-notes:0, mlx-lora-notes:1, mlx-lora-notes:5, mlx-lora-notes:2, retrieval-notes:5
- differs from both parents: false

| fused id | score | dense_rank | lexical_rank |
| --- | --- | --- | --- |
| mlx-lora-notes:0 | 0.032787 | 1 | 1 |
| mlx-lora-notes:1 | 0.032002 | 2 | 3 |
| mlx-lora-notes:5 | 0.032002 | 3 | 2 |
| mlx-lora-notes:2 | 0.031250 | 4 | 4 |
| retrieval-notes:5 | 0.015385 | 5 | absent |

## POST /search

- dense top-5: retrieval-notes:5, fieldguide-runbook:2, retrieval-notes:0, fieldguide-runbook:0, fieldguide-runbook:5
- lexical top-5: fieldguide-runbook:0, fieldguide-runbook:2, fieldguide-runbook:1, fieldguide-runbook:4, fieldguide-runbook:3
- fused top-5: fieldguide-runbook:2, fieldguide-runbook:0, fieldguide-runbook:4, fieldguide-runbook:1, fieldguide-runbook:3
- differs from both parents: true

| fused id | score | dense_rank | lexical_rank |
| --- | --- | --- | --- |
| fieldguide-runbook:2 | 0.032258 | 2 | 2 |
| fieldguide-runbook:0 | 0.032018 | 4 | 1 |
| fieldguide-runbook:4 | 0.029911 | 10 | 4 |
| fieldguide-runbook:1 | 0.029387 | 14 | 3 |
| fieldguide-runbook:3 | 0.029083 | 13 | 5 |

## train.jsonl valid.jsonl test.jsonl

- dense top-5: mlx-lora-notes:2, mlx-lora-notes:5, mlx-lora-notes:4, fieldguide-runbook:4, fieldguide-runbook:0
- lexical top-5: mlx-lora-notes:2, mlx-lora-notes:5, mlx-lora-notes:4
- fused top-5: mlx-lora-notes:2, mlx-lora-notes:5, mlx-lora-notes:4, fieldguide-runbook:4, fieldguide-runbook:0
- differs from both parents: false

| fused id | score | dense_rank | lexical_rank |
| --- | --- | --- | --- |
| mlx-lora-notes:2 | 0.032787 | 1 | 1 |
| mlx-lora-notes:5 | 0.032258 | 2 | 2 |
| mlx-lora-notes:4 | 0.031746 | 3 | 3 |
| fieldguide-runbook:4 | 0.015625 | 4 | absent |
| fieldguide-runbook:0 | 0.015385 | 5 | absent |

## 2-4 sentences

- dense top-5: fieldguide-runbook:2, retrieval-notes:1, retrieval-notes:4, retrieval-notes:5, mlx-lora-notes:3
- lexical top-5: fieldguide-runbook:2, fieldguide-runbook:1
- fused top-5: fieldguide-runbook:2, fieldguide-runbook:1, retrieval-notes:1, retrieval-notes:4, retrieval-notes:5
- differs from both parents: true

| fused id | score | dense_rank | lexical_rank |
| --- | --- | --- | --- |
| fieldguide-runbook:2 | 0.032787 | 1 | 1 |
| fieldguide-runbook:1 | 0.030018 | 12 | 2 |
| retrieval-notes:1 | 0.016129 | 2 | absent |
| retrieval-notes:4 | 0.015873 | 3 | absent |
| retrieval-notes:5 | 0.015625 | 4 | absent |

## Reported difference

Fusion differed on query: POST /search. Top ids: fieldguide-runbook:2, fieldguide-runbook:0, fieldguide-runbook:4, fieldguide-runbook:1, fieldguide-runbook:3.
