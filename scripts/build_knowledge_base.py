import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.ingest import load_markdown_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_DIR = PROJECT_ROOT / "data" / "vectorstore"

OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


def main():

    print("=" * 60)
    print("BUILDING KNOWLEDGE BASE")
    print("=" * 60)

    chunks = load_markdown_documents(
        DOCUMENTS_DIR
    )

    print()
    print(f"[RESULT] Total chunks: {len(chunks)}")

    counts = {}

    serialized_chunks = []

    for index, chunk in enumerate(chunks):

        counts[chunk.board] = (
            counts.get(chunk.board, 0) + 1
        )

        serialized_chunks.append(
            {
                "id": index,
                "text": chunk.text,
                "metadata": {
                    "board": chunk.board,
                    "source": chunk.source,
                    "section": chunk.section,
                },
            }
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            serialized_chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("[CHUNKS BY BOARD]")

    for board, count in sorted(counts.items()):
        print(f"  {board}: {count}")

    print()
    print(f"[OUTPUT] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()