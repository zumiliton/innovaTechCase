from pathlib import Path

import faiss
import json

from app.rag.embeddings import EmbeddingModel


class Retriever:

    def __init__(
        self,
        index_path,
        metadata_path,
        embedding_model_path=None,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index = faiss.read_index(str(self.index_path))

        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.embedding_model = EmbeddingModel(
            model_path=embedding_model_path
            if embedding_model_path
            else "models/embeddings/all-MiniLM-L6-v2"
        )

    def retrieve(
        self,
        question: str,
        board: str,
        top_k: int = 5,
    ) -> list[dict]:

        query_embedding = self.embedding_model.encode(
            [question]
        )

        scores, indices = self.index.search(
            query_embedding,
            len(self.documents),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            document = self.documents[index]

            document_board = (
                document["metadata"]["board"]
            )

            if document_board != board:
                continue

            results.append(
                {
                    "score": float(score),
                    "text": document["text"],
                    "metadata": document["metadata"],
                }
            )

            if len(results) >= top_k:
                break

        return results