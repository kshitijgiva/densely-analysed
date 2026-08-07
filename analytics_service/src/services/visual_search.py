import time
from typing import Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from config import (
    CHROMADB_DATABASE,
    CHROMADB_HOST,
    CHROMADB_PORT,
    CHROMADB_TENANT,
    CHROMADB_VISUAL_COLLECTION,
    VISUAL_SEARCH_TTL_DAYS,
)


class VisualSearchClient:
    """CLIP image-embedding store for semantic ('find people wearing X') search.

    Deliberately separate from ChromaDBClient (services/chroma.py), which
    holds short-TTL OSNet re-id vectors for cross-camera identity matching.
    This collection is longer-lived by design (see config.py note) and also
    stores a small thumbnail per entry so a search result can be displayed.
    """

    def __init__(
        self,
        host: str = CHROMADB_HOST,
        port: int = CHROMADB_PORT,
        tenant: str = CHROMADB_TENANT,
        database: str = CHROMADB_DATABASE,
        collection_name: str = CHROMADB_VISUAL_COLLECTION,
        ttl_days: int = VISUAL_SEARCH_TTL_DAYS,
    ):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            tenant=tenant,
            database=database,
            settings=Settings(anonymized_telemetry=False),
        )
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Connected to ChromaDB visual-search collection at {host}:{port} ({collection_name})")

    @staticmethod
    def _embedding_values(embedding) -> List[float]:
        values = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(values)
        if norm == 0:
            raise ValueError("Cannot store a zero-length embedding")
        return (values / norm).tolist()

    def upsert_visual(
        self,
        person_id: str,
        embedding,
        thumbnail_b64: str,
        store_id: str,
        camera_id: str,
        seen_at: Optional[int] = None,
    ) -> str:
        """One CLIP embedding + thumbnail per visit (not per person - the same
        person on a later visit gets a new entry, since appearance/clothing
        may differ)."""
        seen_at = seen_at or int(time.time())
        vector_id = f"visit:{person_id}"
        self.collection.upsert(
            ids=[vector_id],
            embeddings=[self._embedding_values(embedding)],
            metadatas=[{
                "person_id": person_id,
                "store_id": store_id,
                "camera_id": camera_id,
                "seen_at": seen_at,
                "expires_at": seen_at + self.ttl_seconds,
                "thumbnail_b64": thumbnail_b64,
            }],
        )
        return vector_id

    def search_by_embedding(
        self,
        query_embedding,
        store_id: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """query_embedding must be in the same CLIP space (use visual_embeddings.encode_text)."""
        if self.collection.count() == 0:
            return []

        now = int(time.time())
        conditions = [{"expires_at": {"$gt": now}}]
        if store_id:
            conditions.append({"store_id": {"$eq": store_id}})
        if start_ts:
            conditions.append({"seen_at": {"$gte": start_ts}})
        if end_ts:
            conditions.append({"seen_at": {"$lte": end_ts}})
        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        result = self.collection.query(
            query_embeddings=[self._embedding_values(query_embedding)],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        matches = []
        for vector_id, metadata, distance in zip(
            result["ids"][0], result["metadatas"][0], result["distances"][0]
        ):
            matches.append({
                "vector_id": vector_id,
                "person_id": metadata["person_id"],
                "store_id": metadata["store_id"],
                "camera_id": metadata["camera_id"],
                "seen_at": metadata["seen_at"],
                "similarity": 1.0 - float(distance),
                "thumbnail_b64": metadata.get("thumbnail_b64"),
            })
        return matches

    def purge_expired(self, now: Optional[int] = None) -> int:
        now = now or int(time.time())
        expired = self.collection.get(where={"expires_at": {"$lte": now}}, include=[])
        ids = expired["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)
