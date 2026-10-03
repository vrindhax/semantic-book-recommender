"""Step 4b: Query the persistent Chroma index without re-embedding books."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "books"


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

        query_vector = model.encode(query).tolist()

        # Chroma does the similarity search internally — no manual loop needed.
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3,
        )

        print(f"\nQuery: \"{query}\"")
        print("Top matches:\n")

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            print(f"{rank}. {meta['title']} by {meta['author']} — distance: {dist:.4f}")
            print(f"   {doc}\n")


if __name__ == "__main__":
    main()