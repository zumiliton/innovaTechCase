import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retriever import Retriever


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

    retriever = Retriever(
        index_path=INDEX_FILE,
        metadata_path=METADATA_FILE,
    )

    board = "UNO"

    question = (
        "How many digital input/output pins "
        "does this board have?"
    )

    print()
    print("=" * 60)
    print(f"BOARD: {board}")
    print(f"QUESTION: {question}")
    print("=" * 60)

    results = retriever.retrieve(
        question=question,
        board=board,
        top_k=5,
    )

    for i, result in enumerate(results, start=1):

        print()
        print(f"--- RESULT {i} ---")
        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Source: "
            f"{result['metadata']['source']}"
        )

        print(
            f"Section: "
            f"{result['metadata']['section']}"
        )

        print(
            f"Board: "
            f"{result['metadata']['board']}"
        )

        print()
        print(result["text"])


if __name__ == "__main__":
    main()