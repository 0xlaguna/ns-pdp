# Rerank spot-check

Query: Why not run the cross-encoder on every chunk?

Top id: retrieval-notes:2

> . A cross-encoder scores a pair. You pack the query and one passage into one sequence. Self-attention then mixes the two sides, and the network emits one relevance number for that pair rather than a vector you could file in an index. Change the query and every score is invalid. The passage encoding is not reusable across queries, which is why a cross-encoder is the wrong object to precompute.
