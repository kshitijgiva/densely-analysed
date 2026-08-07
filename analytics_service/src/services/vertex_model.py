import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
import numpy as np

import asyncio
from concurrent.futures import ThreadPoolExecutor

embedding_dimension = 1408

class VertexAIService:
    def __init__(self, project_id: str, location: str = "us-central1"):
        vertexai.init(project=project_id, location=location)
        self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        self.executor = ThreadPoolExecutor()

    async def get_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_image_embedding_sync,
            image_bytes
        )

    def _get_image_embedding_sync(self, image_bytes: bytes) -> np.ndarray:
        image = Image(image_bytes)
        embeddings = self.model.get_embeddings(image=image, dimension=embedding_dimension)
        return np.array(embeddings.image_embedding)
