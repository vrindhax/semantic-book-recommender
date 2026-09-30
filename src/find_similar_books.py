"""Step 2: Embed all books, embed a query, and rank by cosine similarity."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "books.json"


def load_books(path: Path) -> list[dict]:
    """Load the mock book dataset from a JSON file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_book_text(book: dict) -> str:
    """Combine fields into the single string we embed."""
    return f"{book['title']} by {book['author']}. {book['description']}"


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return the cosine of the angle between two vectors.

    Formula: (A . B) / (||A|| * ||B||)
    Result ranges from -1 (opposite) to 1 (identical direction).
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


def main() -> None:
    books = load_books(DATA_PATH)
    print(f"Loaded {len(books)} books.\n")

    model = SentenceTransformer(MODEL_NAME)

    # Embed every book's combined text into a matrix of vectors.
    book_texts = [build_book_text(book) for book in books]
    book_vectors = model.encode(book_texts)  # shape: (num_books, 384)

    # The query we're searching for. Change this and re-run to experiment.
    query = "a lonely robot learning to love"
    query_vector = model.encode(query)  # shape: (384,)

    # Compare the query against every book and store the score.
    results = []
    for book, book_vector in zip(books, book_vectors):
        score = cosine_similarity(query_vector, book_vector)
        results.append((score, book))

    # Sort so the highest similarity score comes first.
    results.sort(key=lambda item: item[0], reverse=True)

    print(f"Query: \"{query}\"\n")
    print("Ranked results (best match first):\n")
    for rank, (score, book) in enumerate(results, start=1):
        print(f"{rank}. {book['title']} — score: {score:.4f}")


if __name__ == "__main__":
    main()