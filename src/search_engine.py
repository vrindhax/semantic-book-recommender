"""Step 3: A reusable semantic search function with an interactive loop."""

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
    """Return the cosine of the angle between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


def search(query: str, model: SentenceTransformer, books: list[dict],
           book_vectors: np.ndarray, top_n: int = 3) -> list[tuple[float, dict]]:
    """Embed a query and return the top_n most similar books.

    Reuses an already-loaded model and already-embedded book_vectors,
    so this function is cheap to call repeatedly.
    """
    query_vector = model.encode(query)

    scored = []
    for book, book_vector in zip(books, book_vectors):
        score = cosine_similarity(query_vector, book_vector)
        scored.append((score, book))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_n]


def print_results(query: str, results: list[tuple[float, dict]]) -> None:
    """Pretty-print search results to the terminal."""
    print(f"\nQuery: \"{query}\"")
    print("Top matches:\n")
    for rank, (score, book) in enumerate(results, start=1):
        print(f"{rank}. {book['title']} by {book['author']} — score: {score:.4f}")
        print(f"   {book['description']}\n")


def main() -> None:
    books = load_books(DATA_PATH)
    print(f"Loaded {len(books)} books.")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Expensive step: embed every book ONCE, before the loop starts.
    book_texts = [build_book_text(book) for book in books]
    book_vectors = model.encode(book_texts)

    print("\nReady. Type a search query, or 'quit' to exit.\n")

    # Interactive loop: cheap to run per query, model/books stay loaded.
    while True:
        query = input("Search> ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not query:
            continue

        results = search(query, model, books, book_vectors, top_n=3)
        print_results(query, results)


if __name__ == "__main__":
    main()