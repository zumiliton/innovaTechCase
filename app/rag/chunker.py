from dataclasses import dataclass
import re


@dataclass
class DocumentChunk:
    text: str
    board: str
    source: str
    section: str


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """
    Split Markdown document by headings.

    Returns:
        [(section_title, section_text), ...]
    """

    pattern = r"^(#{1,6})\s+(.+)$"

    matches = list(re.finditer(pattern, text, re.MULTILINE))

    if not matches:
        return [("Document", text.strip())]

    sections = []

    for i, match in enumerate(matches):
        section_title = match.group(2).strip()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()

        if section_text:
            sections.append(
                (
                    section_title,
                    section_text,
                )
            )

    return sections


def split_large_text(
    text: str,
    max_chars: int = 2500,
    overlap: int = 300,
) -> list[str]:
    """
    Split large text into overlapping chunks.
    """

    if len(text) <= max_chars:
        return [text.strip()]

    chunks = []

    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def chunk_document(
    text: str,
    board: str,
    source: str,
    max_chars: int = 2500,
    overlap: int = 300,
) -> list[DocumentChunk]:
    """
    Convert a Markdown document into structured chunks.
    """

    sections = split_markdown_sections(text)

    chunks = []

    for section_title, section_text in sections:

        section_chunks = split_large_text(
            section_text,
            max_chars=max_chars,
            overlap=overlap,
        )

        for chunk in section_chunks:
            chunks.append(
                DocumentChunk(
                    text=chunk,
                    board=board,
                    source=source,
                    section=section_title,
                )
            )

    return chunks