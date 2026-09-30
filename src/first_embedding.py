"""Step 1: Load a local embedding model and generate our first vector."""

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
    """Combine fields into the single string we embed.

    What you choose to embed strongly affects search quality.
    """
    return f"{book['title']} by {book['author']}. {book['description']}"


def main() -> None:
    books = load_books(DATA_PATH)
    print(f"Loaded {len(books)} books.")

    # First run downloads the model (~90 MB) and caches it locally.
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    first_book = books[0]
    text = build_book_text(first_book)
    vector = model.encode(text)

    print(f"\nText embedded:\n  {text}\n")
    print(f"Vector type:       {type(vector).__name__}")
    print(f"Vector shape:      {vector.shape}")
    print(f"Data type:         {vector.dtype}")
    print(f"First 8 values:    {np.round(vector[:8], 4)}")
    print(f"Vector length (L2 norm): {np.linalg.norm(vector):.4f}")


if __name__ == "__main__":
    main()