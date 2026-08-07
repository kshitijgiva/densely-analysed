import time
from typing import Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from config import (
    CHROMADB_COLLECTION,
    CHROMADB_DATABASE,
    CHROMADB_HOST,
    CHROMADB_PORT,
    CHROMADB_TENANT,
    CHROMADB_TTL_HOURS,
)


class ChromaDBClient:
    """Short-lived OSNet embedding store used for cross-process re-identification."""

    def __init__(
        self,
        host: str = CHROMADB_HOST,
        port: int = CHROMADB_PORT,
        tenant: str = CHROMADB_TENANT,
        database: str = CHROMADB_DATABASE,
        collection_name: str = CHROMADB_COLLECTION,
        ttl_hours: int = CHROMADB_TTL_HOURS,
    ):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            tenant=tenant,
            database=database,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.ttl_seconds = ttl_hours * 60 * 60
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Connected to ChromaDB at {host}:{port} ({collection_name})")

    @staticmethod
    def vector_id(person_id: str) -> str:
        return f"person:{person_id}"

    @staticmethod
    def _embedding_values(embedding) -> List[float]:
        values = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(values)
        if norm == 0:
            raise ValueError("Cannot store a zero-length embedding")
        return (values / norm).tolist()

    def match_identity(
        self,
        embedding,
        store_id: str,
        threshold: float,
        top_k: int = 1,
    ) -> Optional[Dict]:
        """Return the best unexpired match, or None when it is below threshold."""
        if self.collection.count() == 0:
            return None

        now = int(time.time())
        result = self.collection.query(
            query_embeddings=[self._embedding_values(embedding)],
            n_results=top_k,
            where={
                "$and": [
                    {"store_id": {"$eq": store_id}},
                    {"expires_at": {"$gt": now}},
                ]
            },
            include=["metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return None

        distance = float(result["distances"][0][0])
        similarity = 1.0 - distance
        if similarity <= threshold:
            return None

        metadata = result["metadatas"][0][0]
        return {
            "vector_id": result["ids"][0][0],
            "person_id": metadata["person_id"],
            "similarity": similarity,
            "metadata": metadata,
        }

    def upsert_identity(
        self,
        person_id: str,
        embedding,
        store_id: str,
        camera_id: str,
        seen_at: Optional[int] = None,
    ) -> str:
        """Insert or refresh one representative embedding for a person."""
        seen_at = seen_at or int(time.time())
        vector_id = self.vector_id(person_id)
        self.collection.upsert(
            ids=[vector_id],
            embeddings=[self._embedding_values(embedding)],
            metadatas=[{
                "person_id": person_id,
                "store_id": store_id,
                "camera_id": camera_id,
                "last_seen_at": seen_at,
                "expires_at": seen_at + self.ttl_seconds,
            }],
        )
        return vector_id

    def purge_expired(self, now: Optional[int] = None) -> int:
        """Delete embeddings whose application-level TTL has elapsed."""
        now = now or int(time.time())
        expired = self.collection.get(
            where={"expires_at": {"$lte": now}},
            include=[],
        )
        ids = expired["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)
