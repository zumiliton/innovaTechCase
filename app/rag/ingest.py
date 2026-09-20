from pathlib import Path

from app.rag.chunker import DocumentChunk, chunk_document


BOARD_DIRECTORIES = {
    "mega": "MEGA",
    "uno": "UNO",
    "nano": "NANO",
}


def load_markdown_documents(
    documents_dir: str | Path,
) -> list[DocumentChunk]:

    documents_dir = Path(documents_dir)

    all_chunks = []

    for directory_name, board_name in BOARD_DIRECTORIES.items():

        board_dir = documents_dir / directory_name

        if not board_dir.exists():
            print(f"[WARNING] Missing directory: {board_dir}")
            continue

        markdown_files = sorted(board_dir.glob("*.md"))

        print(
            f"[INGEST] {board_name}: "
            f"{len(markdown_files)} markdown files"
        )

        for file_path in markdown_files:

            text = file_path.read_text(
                encoding="utf-8"
            )

            chunks = chunk_document(
                text=text,
                board=board_name,
                source=file_path.name,
            )

            all_chunks.extend(chunks)

            print(
                f"  - {file_path.name}: "
                f"{len(chunks)} chunks"
            )

    return all_chunks