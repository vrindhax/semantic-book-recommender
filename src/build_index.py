"""Step 4a: Build a persistent Chroma vector index from our books."""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "books_real_clean.json"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "books_real"


def load_books(path: Path) -> list[dict]:
    """Load the mock book dataset from a JSON file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_book_text(book: dict) -> str:
    """Combine fields into the single string we embed."""
    return f"{book['title']} by {book['author']}. {book['description']}"


def main() -> None:
    books = load_books(DATA_PATH)
    print(f"Loaded {len(books)} books.")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Persistent client writes to disk at CHROMA_PATH instead of memory-only.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Wipe any old version of this collection so re-running stays clean.
    client.get_or_create_collection(COLLECTION_NAME)
    client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    book_texts = [build_book_text(book) for book in books]
    embeddings = model.encode(book_texts).tolist()  # Chroma wants plain lists

    # Chroma requires each item to have a unique string ID.
    ids = [str(book["id"]) for book in books]

    # Metadata lets us filter later (e.g. by author) without re-embedding.
    metadatas = [
        {"title": book["title"], "author": book["author"], "genre": book["genre"]}
        for book in books
    ]

    # Chroma caps how many items can be added in a single call, so we
    # split everything into smaller batches and add them one at a time.
    BATCH_SIZE = 5000

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=book_texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Added batch {start}–{min(end, len(ids))} of {len(ids)}")

    print(f"Indexed {collection.count()} books into Chroma at '{CHROMA_PATH}'.")


if __name__ == "__main__":
    main()