def build_context(
    results: list[dict],
) -> str:

    context_parts = []

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        context_parts.append(
            f"""
SOURCE {i}
Document: {metadata["source"]}
Section: {metadata["section"]}
Board: {metadata["board"]}

{result["text"]}
""".strip()
        )

    return "\n\n".join(context_parts)