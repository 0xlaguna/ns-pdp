"""One-shot writer for the hand-authored chat rows. Not part of the training path."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"

TRAIN = [
    (
        "What does a bi-encoder store for each passage, and why can that happen before a question arrives?",
        "A bi-encoder runs each passage through the transformer on its own and stores one vector. The passage call does not see the query, so those vectors can be computed once and L2-normalized. A new question encodes only the query. After normalization, cosine similarity is an inner product.",
    ),
    (
        "Why is a cross-encoder the wrong thing to precompute over a corpus?",
        "A cross-encoder packs the query and one passage into one sequence and emits a single relevance score for that pair. Self-attention mixes the two sides, so the passage side is not a vector you can file in an index. Change the query and every stored score would be invalid.",
    ),
    (
        "Why does this pipeline run the cross-encoder only on a shortlist?",
        "The joint forward pass is the expensive part, so a retrieve step nominates about twenty fused hits first. The cross-encoder scores only those candidates and the caller keeps five. Scoring every chunk would repeat a full pairwise pass per chunk, which is the wrong shape even for a handful of notes.",
    ),
    (
        "What does Recall@k report, and what can a rerank still not repair?",
        "Recall@k is 1 when a labeled chunk id shows up anywhere in the top k, and 0 when it does not. A rerank can move a relevant id upward inside the list it was given. It cannot recover an id the retriever dropped. If the shortlist already missed, the cross-encoder never sees the right text.",
    ),
    (
        "How is a chunk id built, and why must retrieval-notes:0 stay stable?",
        "The id is the filename stem, a colon, and the window index starting at 0. retrieval-notes:0 has to mean the same string after a rebuild. If that string changes, a later labeled set is just a guess about which window it named.",
    ),
    (
        "What window width and overlap does this corpus use, and how does a cut choose a word boundary?",
        "The default width is 800 characters with an overlap of 120. The end of a window snaps back to a space in the last 80 characters when one exists, so a cut prefers a word boundary. Overlap keeps a sentence that straddles a cut inside two neighboring chunks.",
    ),
    (
        "What does GET /health return, and which port serves it?",
        "The retrieval service listens on port 8000. Liveness is GET /health, and the body is a status of ok plus the number of chunks currently loaded. A healthy check is a chunk count that matches the JSON lines file.",
    ),
    (
        "What does POST /search accept, and what fields does each hit carry?",
        "Search is POST /search. The request carries a query string and a k, and the response carries hits. Each hit has an id, the chunk text, and a rerank_score from the second-stage scorer. Clients send that shape, and the service does not rewrite the corpus while answering.",
    ),
    (
        "What does the fusion constant k do, and how is it different from the k on POST /search?",
        "Reciprocal rank fusion uses a constant k of 60. The score added for a list is one divided by k plus the rank, and ranks start at 1. That constant is a rank smoother, not a cosine. The request k is how many hits to return.",
    ),
    (
        "What score does a document get when it is missing from one of the two ranked lists?",
        "A document missing from one list does not receive a term from that list. Ties in the fused score break by id. The fusion constant only shapes the combined rank, and it is not the k on the search request.",
    ),
    (
        "Why do retrieval, fine-tuning, and speech each get their own virtualenv?",
        "Run the retrieval code from its own virtualenv. The fine-tune environment and the speech environment are separate, because those dependency sets do not solve cleanly together. Model weights, adapters, and the vector index stay on disk and out of git.",
    ),
    (
        "What must a failed index rebuild leave untouched?",
        "When a job rebuilds the index, a failed job must leave the previous chunk file and the previous index in place. A path that escapes the raw notes directory is a failed job, not a search. POST /search does not rewrite the corpus. Rebuilding the chunk file is an offline step.",
    ),
    (
        "How long should a labeled assistant turn be, and why is that band used?",
        "A labeled assistant turn is 2 to 4 sentences. That band is long enough to state the fact and the reason, and short enough that a small model can learn the rhythm. The same band is what we want once a generator is wired to search: ground the reply in a returned hit, then stop.",
    ),
    (
        "Which ports do the token stream, the voice socket, and the static page use?",
        "The token service listens on port 8001. Its chat-completions route is a server-sent event stream: each event line carries a choice with a delta content string, and the stream ends with a done sentinel. The voice process listens on port 8010 at the WebSocket path /ws, and the static browser page is served on port 5500. These processes bind to loopback.",
    ),
    (
        "What does mlx_lm.lora train, and what does it refuse to duplicate?",
        "The command is mlx_lm.lora. It trains a low-rank adapter on top of a frozen language model, leaving the base matrix where it is and learning a pair of small matrices whose product is the update. You store the adapter, not a second full copy of the network. Rank is the width of that update.",
    ),
    (
        "When does this training command take the QLoRA path instead of ordinary LoRA?",
        "QLoRA keeps the base weights in low bit width, here a 4-bit checkpoint, and keeps the adapter matrices in higher precision so the optimizer has a stable update. mlx_lm.lora takes the QLoRA path when the checkpoint you pass is already quantized. Point the same command at a full-precision model and the run is ordinary LoRA. This project does not quantize inside the training loop.",
    ),
    (
        "Which instruct checkpoint is the default, and what replaces it if a license gate blocks the download?",
        "The checkpoint we train is a 4-bit Llama 3.2 instruct model around one billion parameters. If a license gate blocks that download, the swap is a 4-bit Qwen 2.5 instruct model at about half a billion parameters, and the hyperparameters stay put. The swap, if it happens, is written down. The token used to accept a license stays outside the repo.",
    ),
    (
        "Why is the data argument a directory, and what is each of the three jsonl files for?",
        "The data argument is a directory, not a single file. train.jsonl supplies the updates. If valid.jsonl is in that directory, mlx_lm.lora reports validation loss during the run, and test.jsonl is what evaluate mode reads when the test flag is set.",
    ),
    (
        "What does one chat example look like in this set, and why is chat the format we keep?",
        "Each line is one JSON object, and a single example must not be split across lines. The chat shape is a messages list with a user turn and then an assistant turn, each carrying a role and content. This set uses chat so the model's own chat template can wrap the turns. Keys the loader does not expect are ignored, so extra fields would not enter the loss.",
    ),
    (
        "What batch size does the mlx-lm guide default to, and what does this project use when memory is tight?",
        "The documented default batch size is 4. This project uses one sequence at a time. The number of layers that receive adapters can be reduced, which shrinks the backward pass, and that cut is the documented response to an out-of-memory error. Sequence length is capped so one long note cannot blow the activation memory.",
    ),
    (
        "Why will 200 steps on this set not settle which LoRA rank is better in general?",
        "The iteration count is a fixed budget, not an epoch count, so a small hand-written set still finishes in one sitting. Two hundred steps will not support a claim about which rank is universally better. They will show whether the loss moved, and whether rank 16 moved it differently from rank 8.",
    ),
    (
        "Where do the learned adapter weights live, and what is a resume supposed to point at?",
        "The adapter directory is the artifact. Learned weights land on disk and stay out of git. A resume points at the saved adapter file. Fusing the adapter back into the base is a separate command and is not part of the training log.",
    ),
    (
        "What has to be in the training log, and which side wins if the command line disagrees with the yaml?",
        "The log must contain the iteration losses, so a parser can read the first train loss, the last train loss, and the last validation loss when valid.jsonl was present. The same data directory, the same seed, and the same flags are reused when the second run changes rank. If the command line and the YAML disagree, the command line wins, which is why the exact invocation is written next to the log.",
    ),
    (
        "Why does this fine-tune stay on the host instead of inside the training container?",
        "Training stays on the host. A container on this plan does not expose Metal, so mlx_lm.lora is not something we expect to run inside that image. The quantized base plus the higher-precision adapter is the QLoRA setup the guide describes, and the three jsonl files are the whole supervised set.",
    ),
]

VALID = [
    (
        "Once passage vectors are L2-normalized, what does a cosine comparison become?",
        "After normalization, cosine similarity is an inner product, and a flat index can scan it by brute force while the collection is small. That independence is the operational point of a bi-encoder: the expensive passage encoding happens offline. A near paraphrase can still outrank the answering sentence, because query tokens never attend to passage tokens.",
    ),
    (
        "What is the limit of encoding the query and the passage in two separate calls?",
        "Tokens in the query never attend to tokens in the passage, so a near paraphrase can outrank the sentence that actually answers the question. A cross-encoder repairs the order by scoring the query and one passage as one sequence. Those pair scores cannot be computed once and reused for the next question.",
    ),
    (
        "When does the retrieval service load its indexes, and does a search request rebuild the chunk file?",
        "The service loads its indexes once at startup. Rebuilding the chunk file is an offline step. POST /search does not rewrite the corpus.",
    ),
    (
        "How should a blank metric in the activity log be filled?",
        "A blank metric stays blank until the command has been run. Copy the number from the terminal. Do not refill it from memory on a later day. Latency and recall belong in the eval folder as JSON before they are copied into the log.",
    ),
    (
        "What does prompt masking change about the loss on a chat row?",
        "Prompt masking, when it is on, drops the user tokens out of the loss and trains on the assistant span. For chat data the final message is the completion. Sequence length is capped so one long note cannot blow the activation memory.",
    ),
    (
        "Which other row shapes will the loader accept, and what happens to keys it does not expect?",
        "A prompt-plus-completion line and a raw text line are also legal shapes. This set uses chat so the model's own chat template can wrap the turns. Keys the loader does not expect are ignored, so labels hidden in extra fields will not enter the loss.",
    ),
]

TEST = [
    (
        "Which vectors are computed ahead of time, and which ones are computed when a question arrives?",
        "Passage vectors from the bi-encoder are what the local dense index stores. Query vectors are the only ones computed at search time. The cross-encoder never writes those vectors. It only re-scores pairs that the earlier stage already returned.",
    ),
    (
        "What does a healthy retrieval check look like on /health and on /search?",
        "GET /health returns ok and a chunk count that matches the JSON lines file. POST /search for a question these notes can answer returns a hit whose id prefix is the note that contains the fact, and the hit text actually contains that fact. Latency and recall belong in the eval folder as JSON before they are copied into the activity log.",
    ),
    (
        "In the planned rank comparison, what is held fixed so a loss gap can be blamed on rank?",
        "Rank is the width of the low-rank update. A later comparison changes rank and leaves the data, the seed, and the step count alone, so a loss gap can be attributed to rank. You store the adapter, not a second full copy of the network. Two hundred steps still will not support a claim about which rank is universally better.",
    ),
    (
        "Where is a Hugging Face token allowed to live when a download asks for a login?",
        "If a download asks for a Hugging Face login, the token stays in the local credential store and never in a committed file. Notes, JSON lines, and metric files stay in the tree. Weights, adapters, and the vector index stay on disk and out of git.",
    ),
    (
        "Which three files are the whole supervised set, and why is the training process not the container?",
        "The three files train.jsonl, valid.jsonl, and test.jsonl are the whole supervised set. train.jsonl supplies the updates, valid.jsonl puts a validation loss in the log, and test.jsonl stays held out for perplexity after training. Training stays on the host because a container on this plan does not expose Metal.",
    ),
    (
        "What two constraints does the chunker place on overlap and on an empty note?",
        "Overlap must be smaller than the window. An empty note contributes no rows. The end of a window snaps back to a space in the last 80 characters when one exists, so a cut prefers a word boundary. The default width is 800 characters with an overlap of 120.",
    ),
]


def dump(name: str, rows: list[tuple[str, str]]) -> None:
    path = ROOT / f"{name}.jsonl"
    lines = []
    for user, assistant in rows:
        lines.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    dump("train", TRAIN)
    dump("valid", VALID)
    dump("test", TEST)


if __name__ == "__main__":
    main()
