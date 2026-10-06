"""Step 6: Streamlit web UI for the semantic book recommender."""

from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "books_real"


@st.cache_resource
def load_model() -> SentenceTransformer:
    """Load the embedding model once per server session, not per request."""
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_collection():
    """Connect to the persistent Chroma collection once per server session."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(COLLECTION_NAME)


def main() -> None:
    st.set_page_config(page_title="Semantic Book Recommender", page_icon="📚")
    st.title("📚 Semantic Book Recommender")
    st.caption("Search by meaning, not just keywords. Powered by local embeddings.")

    model = load_model()
    collection = load_collection()

    all_genres = sorted({
        meta["genre"] for meta in collection.get()["metadatas"]
    })

    query = st.text_input(
        "What kind of story are you in the mood for?",
        placeholder="e.g. a lonely robot learning to love",
    )
    genre_filter = st.selectbox("Filter by genre (optional)", ["Any"] + all_genres)
    top_n = st.slider("Number of results", min_value=1, max_value=6, value=3)

    if not query:
        st.info("Type a query above to get recommendations.")
        return

    query_vector = model.encode(query).tolist()

    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": top_n,
    }
    if genre_filter != "Any":
        query_kwargs["where"] = {"genre": genre_filter}

    results = collection.query(**query_kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        st.warning("No books matched that filter.")
        return

    st.subheader("Top matches")
    for rank, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        similarity = 1 - dist  # convert Chroma's distance back to similarity
        with st.container(border=True):
            st.markdown(f"**{rank}. {meta['title']}** by {meta['author']}")
            st.caption(f"{meta['genre']} · similarity: {similarity:.2f}")
            st.write(doc)


if __name__ == "__main__":
    main()