import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index = None
        self.documents = []

    def build(
        self,
        embeddings,
        documents,
    ):
        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.documents = documents

    def save(self):
        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.documents,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[VECTOR STORE] Saved index: "
            f"{self.index_path}"
        )

        print(
            f"[VECTOR STORE] Saved metadata: "
            f"{self.metadata_path}"
        )