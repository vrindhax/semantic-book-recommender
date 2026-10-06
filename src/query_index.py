"""Step 4b: Query the persistent Chroma index without re-embedding books."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "books_real"


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Connected to collection with {collection.count()} books.\n")

    print("Ready. Type a search query, or 'quit' to exit.\n")

    while True:
        query = input("Search> ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not query:
            continue

        genre_filter = input("Genre (press Enter to skip)> ").strip()

        query_vector = model.encode(query).tolist()

        # Build the query kwargs dynamically so we only filter when asked.
        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": 3,
        }
        if genre_filter:
            query_kwargs["where"] = {"genre": genre_filter}

        results = collection.query(**query_kwargs)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        print(f"\nQuery: \"{query}\"" + (f" | Genre: {genre_filter}" if genre_filter else ""))
        print("Top matches:\n")

        if not documents:
            print("No books matched that filter.\n")
            continue

        for rank, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            print(f"{rank}. {meta['title']} by {meta['author']} ({meta['genre']}) — distance: {dist:.4f}")
            print(f"   {doc}\n")

if __name__ == "__main__":
    main()