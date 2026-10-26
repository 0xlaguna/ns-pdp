# Fieldguide demo

Fieldguide is local notes, hybrid search, a small QLoRA adapter, a containerized ingest path, and a websocket voice front.

## Commands

Redis, if it is not already up:

```bash
docker run -d --name fieldguide-redis -p 6379:6379 redis:7-alpine
```

RAG API on port 8000, from `evidence/rag`:

```bash
.venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Ingest worker, from `evidence/rag`:

```bash
REDIS_URL=redis://127.0.0.1:6379 .venv/bin/arq worker.WorkerSettings
```

Token stream on port 8001, from `evidence/cloud/src`:

```bash
FIELDGUIDE_LLM=mlx ../../finetune/.venv/bin/uvicorn stream_server:app --host 127.0.0.1 --port 8001
```

Voice server on port 8010, from `evidence/voice/src`:

```bash
../.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8010
```

Client on port 5500, from `evidence/voice/client`:

```bash
../.venv/bin/python -m http.server 5500
```

Open `http://127.0.0.1:5500/` or send `samples/cross.wav` as an `audio` message.

## Numbers

Recall@5 from https://github.com/0xlaguna/ns-pdp/blob/main/rag/eval/results.json (n=8): dense 0.875, bm25 0.75, hybrid 1.0, hybrid_rerank 1.0.

Both QLoRA runs used 200 steps. Validation loss was 3.507 at rank 8 and 3.608 at rank 16. Run A is the lower one. The adapter loaded by the stream is `evidence/finetune/adapters/run-a`.

Cloud URL: none. `evidence/cloud/notes/deploy.md` records the blocker: the Modal CLI is not installed and no account was authenticated.

Voice medians on `samples/cross.wav`, milliseconds, from https://github.com/0xlaguna/ns-pdp/blob/main/voice/bench/turn_latency.json: vad 20.05708299111575, stt 373.74787498265505, search 150.94708290416747, ttft 616.099665989168, first_wav 1543.8500000163913.

## This run

`samples/cross.wav` transcribed as `What is a cross, encoder, and why is it slow?` The spoken answer was: "The passage is a query, and the cross-encoder scores only those candidates the caller keeps. The passage is not a vector you could file in an index, so the right shape is a shortlist." The socket returned 2 audio messages.

## Limits

Eight labeled questions. Twenty-four training rows. One worker. CPU whisper and Piper. The retrieval image build did not finish, so the API and the worker ran on the host. Redis ran in Docker.
