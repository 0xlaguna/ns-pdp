# Base vs rank-8 adapter

Chosen run: A. It had the lower validation loss (3.507 vs 3.608).
Command shape: `mlx_lm.generate --model mlx-community/Llama-3.2-1B-Instruct-4bit --max-tokens 80 --temp 0 --prompt "<user>"`, then the same command with `--adapter-path adapters/run-a`.
`generate()` in mlx-lm 0.32.0 does not take `temp`. The CLI builds `make_sampler(temp)` and passes `sampler`. This script does the same thing, with temperature 0.
The weights load twice, once for the base model and once with the adapter, instead of twelve times.

## Prompt 1

Which vectors are computed ahead of time, and which ones are computed when a question arrives?

### Base

I can provide a general overview of how vector computations are typically handled in various AI and ML frameworks.

**Vector Computations:**

1. **Linear Algebra:** Many linear algebra operations, such as matrix multiplication, are typically computed ahead of time. This is because these operations can be performed on large datasets, and the results can be stored in memory.
2. **Convolutional Neural Networks (CNN

### Adapter

A vector is a sequence of numbers that can be added or subtracted. The passage does not mention a query, so a new vector cannot be computed once a question arrives. The passage's flow is a failed search, and the passage does not expect a return hit.

## Prompt 2

What does a healthy retrieval check look like on /health and on /search?

### Base

In Google, a healthy "Retrieval" check on the /health page and a healthy "Search" check on the /search page would look like this:

**/health**

* Retrieval: 80% (This indicates that Google's search engine is able to retrieve a large number of relevant results from the web.)
* Crawl Rate: 80% (This indicates that Google's

### Adapter

The retrieval score is the amount of text a chunk id carries, and the score increases by 1000 from the last full-id we knew of. The fusion constant only shapes the combined rank, and is not the k on the search request.

## Prompt 3

In the planned rank comparison, what is held fixed so a loss gap can be blamed on rank?

### Base

In the planned rank comparison, the following are held fixed to prevent a loss gap from being blamed on rank:

1. **Initial Rank**: The initial rank is held fixed to prevent the loss gap from being blamed on the rank.

2. **Initial Score**: The initial score is also held fixed to prevent the loss gap from being blamed on the score.

3. **Initial Rank and Score**: The

### Adapter

The fusion constant k is the k on the top of the combined rank chart, and the score added for a failed search is one divided by k plus the rank, and ranks start at 1. That constant is not a rank itself, so a loss can be blamed on that constant.

## Prompt 4

Where is a Hugging Face token allowed to live when a download asks for a login?

### Base

I can provide you with some general information on this topic.

A Hugging Face token is a type of AI model, specifically a pre-trained language model. When a download asks for a login, it's likely referring to the process of authenticating the user with a token.

In this context, I'm assuming you're asking about how a Hugging Face token can be used to authenticate a user.

### Adapter

The token is a directory, not a single file. A Hugging Face token's directory is the artifact directory, and the login is the same string after a rebuild. The token's metadata is in that string, and the token itself is a directory, not a single file. A new login is not a failed job, but a failed search.

## Prompt 5

Which three files are the whole supervised set, and why is the training process not the container?

### Base

I can provide you with the general structure of a supervised machine learning workflow. However, please note that the specifics may vary depending on the project and the framework being used.

A typical supervised machine learning workflow consists of the following three files:

1. **train.csv**: This file contains the training data, typically in the form of a matrix where each row represents a sample and each column represents a feature.

### Adapter

The three files are the Keras environment and the two jsonl files for the supervised set. The container is a string, and the wrong shape is a 4-bit Llama 3.2 instruct model around one billion parameters.

## Prompt 6

What two constraints does the chunker place on overlap and on an empty note?

### Base

The chunker, a tool for writing poetry, places the following two constraints on overlap and on an empty note:

1. **Overlap**: The chunker places an "over" constraint on overlap. This means that if a line has an "over" constraint on an "empty note", then the line must have an "empty note" constraint on the line below it.

2. **Empty Note

### Adapter

The tokenizer uses a chunker to split the input text into chunks currently, there is no third constraint about an empty note. Overlap is the fact that a sentence contains a word that overlaps with a sentence that ends with a question mark. The passage does not end with a question mark, so that is not a problem.

## What I will actually claim

Closer: "What does a healthy retrieval check look like on /health and on /search?" The adapter says the fusion constant is not the k on the search request. That sentence is in the runbook. The base answer invents Google crawl-rate percentages and never mentions the route contract.

Worse: "In the planned rank comparison, what is held fixed so a loss gap can be blamed on rank?" The notes hold the data, the seed, and the step count fixed. The adapter answers with the fusion constant instead. The base stays on the rank-comparison question, even though it names the wrong things to hold fixed.

Both fail: "Where is a Hugging Face token allowed to live when a download asks for a login?" The notes put the token in the local credential store and keep it out of the repo. The base calls the token a language model. The adapter calls it a directory. Neither answer is the rule.

Train loss on run A ended at 0.038. These six answers are still not the notes. A lower train loss did not make the held-out answers right.
