import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "chunks.json"
)

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "index.faiss"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "metadata.json"
)


def main():

    print("=" * 60)
    print("BUILDING VECTOR STORE")
    print("=" * 60)

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        documents = json.load(f)

    print(
        f"[INPUT] Loaded {len(documents)} chunks"
    )

    texts = [
        document["text"]
        for document in documents
    ]

    print(
        "[EMBEDDINGS] Loading model..."
    )

    embedding_model = EmbeddingModel()

    print(
        "[EMBEDDINGS] Generating embeddings..."
    )

    embeddings = embedding_model.encode(
        texts
    )

    print(
        f"[EMBEDDINGS] Shape: {embeddings.shape}"
    )

    vector_store = VectorStore(
        index_path=INDEX_FILE,
        metadata_path=METADATA_FILE,
    )

    vector_store.build(
        embeddings=embeddings,
        documents=documents,
    )

    vector_store.save()

    print()
    print("DONE")


if __name__ == "__main__":
    main()