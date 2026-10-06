# Semantic Book Recommender

A search engine that finds books by *meaning*, not keywords — powered by local sentence embeddings and a persistent vector database. Search "a lonely robot learning to love" and it finds *Klara and the Sun*, even though none of those words appear in the book's description.

🔗 Live demo: *(add Streamlit Cloud link here once deployed, or a screen recording)*

## What it does

- Takes a natural-language query ("grief and learning to let go") and ranks a dataset of ~10,000 real books by semantic similarity
- Optionally filters results by genre, combining exact metadata filtering with fuzzy semantic ranking
- Runs entirely locally — no API keys, no cost, no data leaving your machine

## How it works

1. **Embedding** — each book's title, author, and description is converted into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`, a free, local Hugging Face model.
2. **Indexing** — vectors are stored in [Chroma]([https://www.trychroma.com/](https://www.trychroma.com/)), a persistent local vector database, along with metadata (title, author, genre) for filtering.
3. **Querying** — a user's search text is embedded the same way, then compared against every book vector using cosine similarity. The closest matches win.
4. **Interface** — a Streamlit app wraps the search in a simple, shareable web UI.

## Tech stack

| Layer | Tool | Why |

|---|---|---|

| Embeddings | `sentence-transformers` (MiniLM-L6-v2) | Free, local, no API key, CPU-friendly |

| Vector storage | ChromaDB (persistent, local) | No server to manage, supports metadata filtering |

| Data processing | pandas | Cleaning a real 10k-row dataset with missing/malformed fields |

| UI | Streamlit | Fast way to turn a Python script into a shareable web app |

## Project evolution

This project was built in deliberate stages, each one adding a real engineering capability:

1. Generate a single embedding, understand what a vector actually is
2. Implement cosine similarity manually with NumPy, to understand the math before trusting a library with it
3. Wrap search into a reusable, testable function
4. Replace manual search with a real persistent vector database (Chroma)
5. Add metadata filtering — combining exact (genre) and fuzzy (semantic) search
6. Build a Streamlit UI
7. Replace the 6-book mock dataset with a real, messy, 10,000-book dataset (cleaning stringified lists, handling missing descriptions, batching inserts around Chroma's size limits)

## Known limitations

Semantic search is probabilistic, not exact — it will confidently return a "best match" even when nothing in the dataset is actually a good fit. Two honest examples found during testing:

- **Negative similarity scores still get shown.** Cosine similarity ranges from -1 to 1; a result scoring -0.39 means the query and book are only weakly related, but the current UI still presents it as the "#1 match." A production version should set a minimum similarity threshold and show "no good match found" instead of forcing out a top-3 list regardless of actual quality.
- **Combining a narrow genre filter with a weakly-related query produces poor results.** Searching "a Japanese book" filtered to the "self-help" genre returned *The Secret* — a book that's self-help, but not Japanese — because the filter eliminated every genuinely relevant option before similarity scoring even ran. This illustrates the trade-off between deterministic filters and semantic ranking discussed in the project's design.

## Running locally

`\`bash

git clone [https://github.com/vrindhax/semantic-book-recommender.git](https://github.com/vrindhax/semantic-book-recommender.git)

cd semantic-book-recommender

python -m venv .venv

.venv\Scripts\activate   # Windows

pip install -r requirements.txt

python src/prepare_[data.py](http://data.py)   # clean the raw dataset (if not already present)

python src/build_[index.py](http://index.py)    # build the Chroma vector index

streamlit run src/[app.py](http://app.py)     # launch the web UI

`\`

## What I'd improve next

- Add a minimum-similarity threshold so weak matches aren't presented with false confidence
- Support multi-genre filtering instead of collapsing each book to a single primary genre
- Add a reranking step (e.g. cross-encoder) to improve top-3 precision
- Deploy publicly on Streamlit Community Cloud

