from pathlib import Path

from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_path: str = "models/embeddings/all-MiniLM-L6-v2",
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found: {model_path}"
            )

        self.model = SentenceTransformer(str(model_path))

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )