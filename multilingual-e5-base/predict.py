# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from typing import List
import json


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-base", cache_folder="cache_models"
        )

    def predict(
        self,
        texts: str = Input(
            default='["In the water, fish are swimming.", "Fish swim in the water.", "A book lies open on the table."]',
            description='text to embed, formatted as JSON list of strings (e.g. ["hello", "world"])',
        ),
        batch_size: int = Input(
            default=32,
            description="Batch size to use when processing text data.",
            ge=0,
        ),
        normalize_embeddings: bool = Input(
            default=True,
            description="Whether to normalize embeddings.",
        ),
    ) -> List[List[float]]:
        input_texts = json.loads(texts)
        embeddings: ndarray = self.model.encode(
            input_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
        return embeddings.tolist()
