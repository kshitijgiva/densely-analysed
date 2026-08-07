import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import chromadb.errors
from config import CHROMADB_DATABASE, CHROMADB_HOST, CHROMADB_PORT, CHROMADB_TENANT

class ChromaDBClient:
    def __init__(
        self, 
        host: str = CHROMADB_HOST, 
        port: int = CHROMADB_PORT,
        tenant: str = CHROMADB_TENANT,
        database: str = CHROMADB_DATABASE
    ):
        # Use the HTTP client to connect to the ChromaDB server running in Docker
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings()
        )
        print(f" connected to {host}:{port}, tenant: {tenant}, database: {database}")
        # Test the connection
        # If you need to set tenant/database, uncomment and adjust as needed
        self.client.set_tenant(tenant)
        self.client.set_database(database)

    def get_collection(self, collection_name: str):
        """Retrieve or create a collection by name."""
        return self.client.get_or_create_collection(collection_name)

    def search_embeddings(self, collection_name: str, query_embedding: List[float], top_k: int, metadata_filter: dict = None):
        """Search embeddings in a specific collection."""
        collection = self.get_collection(collection_name)
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["metadatas", "distances"]
        }
        if metadata_filter:
            query_args["where"] = metadata_filter  # or "filter", depending on your DB

        return collection.query(**query_args)

    def get_metadata_by_ids(self, collection_name: str, ids: List[str]):
        """Fetch metadata for a list of IDs from a specific collection."""
        collection = self.get_collection(collection_name)
        return collection.get(ids=ids, include=["metadatas"])

    def get_embeddings_by_ids(self, collection_name: str, ids: List[str]):
        """Fetch embeddings by IDs from a specific collection."""
        collection = self.get_collection(collection_name)
        return collection.get(ids=ids, include=["embeddings"])

    def get_item_embeddings(self, collection_name: str, skus: List[str]):
        """Fetch embeddings and metadata by item_ids if item_id is used as ID."""
        collection = self.get_collection(collection_name)
        return collection.get(ids=skus, include=["embeddings", "metadatas"])


    def insert_embeddings(self, collection_name: str, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        """
        Insert embeddings into the specified collection.
        Raises an Exception if the insertion fails.
        """
        try:
            collection = self.get_collection(collection_name)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            raise Exception(f"Failed to insert embeddings into collection '{collection_name}': {e}")


    def update__embeddings(self, collection_name: str, ids: List[str], embeddings: Optional[List[List[float]]] = None, metadatas: Optional[List[Dict]] = None):
        """
        Updates existing embeddings and/or metadata in the specified collection.
        """
        collection = self.get_collection(collection_name)
        collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def reset_collection(self, collection_name: str):

        print(f"all the daatabases {self.client.database}")
        print(f"all the collections {self.client.list_collections()}")
        existing_collections = self.client.list_collections()
        if len(existing_collections) > 0 and hasattr(existing_collections[0], "name"):
            existing_names = [col.name for col in existing_collections]
        else:
            existing_names = existing_collections

        if collection_name in existing_names:
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted collection: {collection_name}")
            except chromadb.errors.NotFoundError:
                print(f"Collection {collection_name} not found when deleting - ignoring")

        print(f"Creating collection: {collection_name}")
        collection = self.client.create_collection(collection_name)
        print(f"Created collection: {collection}")

        return collection
