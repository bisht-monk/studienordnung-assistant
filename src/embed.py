import json

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNKS_PATH = "data/processed/chunks.json"
DB_PATH = "data/chroma_db"
COLLECTION_NAME = "modulkatalog"


def build_index():
    # 1. Load the chunks we saved on Day 3
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # 2. Load the embedding model (downloads ~470 MB on first run)
    model = SentenceTransformer(MODEL_NAME)

    # 3. Turn every chunk into a vector.
    # This model expects the prefix "passage: " on documents.
    texts = ["passage: " + c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # 4. Store everything in Chroma (saved on disk in data/chroma_db)
    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete the old collection so re-running doesn't create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    # Add in batches to stay under Chroma's size limits
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Stored {collection.count()} chunks in Chroma at {DB_PATH}")


if __name__ == "__main__":
    build_index()