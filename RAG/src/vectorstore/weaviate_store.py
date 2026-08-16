"""
Weaviate Cloud vector store implementation.
"""
import os
import yaml
from pathlib import Path
from typing import Any
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery, Filter
from dotenv import load_dotenv

class WeaviateVectorStore:
    """
    Handles connection, document insertion, and similarity search in Weaviate Cloud.
    This layer is strictly for storage and retrieval primitives, keeping it agnostic to higher-level RAG logic.
    """

    def __init__(self) -> None:
        """
        Initialize the Weaviate vector store and connect to the cluster.
        Reads credentials from environment variables.
        """
        load_dotenv()
        
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        
        if not weaviate_url or not weaviate_api_key:
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in environment variables.")

        # connect_to_weaviate_cloud expects a WCS cluster URL and Auth Api Key
        # Set a connection timeout configuration to prevent the process from hanging indefinitely
        # if DNS resolution fails or the cluster is temporarily unreachable.
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            skip_init_checks=True,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=10, query=60, insert=120)
            )
        )
        
        self.collection_name = self._load_collection_name()

    def _load_collection_name(self) -> str:
        """
        Load the collection name from the config.yaml file.
        
        Returns:
            str: The collection name.
        """
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if not config_path.exists():
            return "RagDocuments"
            
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            
        return config.get("collection_name", "RagDocuments")

    def delete_collection(self) -> None:
        """
        Delete the vector collection.
        """
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

    def create_collection(self) -> None:
        """
        Create the vector collection if it does not already exist.
        """
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="chunk_id", data_type=DataType.TEXT),
                ],
                # We are providing our own embeddings externally, so we disable the internal vectorizer
                vector_config=Configure.Vectors.self_provided()
            )

    def is_file_ingested(self, source: str) -> bool:
        """
        Check if a file source has already been ingested in Weaviate.
        
        Args:
            source (str): The full path or URL identifier of the document.
            
        Returns:
            bool: True if already ingested, False otherwise.
        """
        if not self.client.collections.exists(self.collection_name):
            return False
            
        collection = self.client.collections.get(self.collection_name)
        response = collection.query.fetch_objects(
            filters=Filter.by_property("source").equal(source),
            limit=1
        )
        return len(response.objects) > 0

    def add_documents(self, texts: list[str], metadatas: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        """
        Insert chunked documents with metadata and embeddings into the collection.
        
        Args:
            texts (list[str]): List of document chunks.
            metadatas (list[dict[str, Any]]): List of metadata dictionaries for each chunk.
            embeddings (list[list[float]]): List of embedding vectors for each chunk.
            
        Raises:
            ValueError: If the lengths of texts, metadatas, and embeddings do not match.
        """
        if not (len(texts) == len(metadatas) == len(embeddings)):
            raise ValueError("Lengths of texts, metadatas, and embeddings must be equal.")
            
        collection = self.client.collections.get(self.collection_name)
        
        with collection.batch.dynamic() as batch:
            for text, metadata, vector in zip(texts, metadatas, embeddings):
                properties = {
                    "text": text,
                    "source": str(metadata.get("source", "")),
                    "filename": str(metadata.get("filename", "")),
                    "title": metadata.get("title") if metadata.get("title") else None,
                    "chunk_id": str(metadata.get("chunk_id", ""))
                }
                
                page = metadata.get("page")
                if page is not None:
                    try:
                        properties["page"] = int(page)
                    except ValueError:
                        pass
                        
                batch.add_object(
                    properties=properties,
                    vector=vector
                )

    def similarity_search(self, query_vector: list[float], limit: int = 4) -> list[dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query_vector (list[float]): The embedding vector of the query.
            limit (int): Maximum number of results to return.
            
        Returns:
            list[dict[str, Any]]: A list of dictionaries representing the search results,
                                  including text, metadata, and distance.
            
        Raises:
            ValueError: If the query_vector is empty.
        """
        if not query_vector:
            raise ValueError("query_vector cannot be empty.")
            
        collection = self.client.collections.get(self.collection_name)
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )
        
        results = []
        for obj in response.objects:
            results.append({
                "text": obj.properties.get("text"),
                "metadata": {
                    "source": obj.properties.get("source"),
                    "filename": obj.properties.get("filename"),
                    "page": obj.properties.get("page"),
                    "title": obj.properties.get("title"),
                    "chunk_id": obj.properties.get("chunk_id"),
                },
                "distance": obj.metadata.distance if obj.metadata else None
            })
            
        return results

    def keyword_search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query (str): The keyword search query.
            limit (int): Maximum number of results to return.
            
        Returns:
            list[dict[str, Any]]: A list of dictionaries representing the search results,
                                  including text, metadata, and BM25 score.
            
        Raises:
            ValueError: If the query is empty.
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")
            
        collection = self.client.collections.get(self.collection_name)
        
        response = collection.query.bm25(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(score=True)
        )
        
        results = []
        for obj in response.objects:
            results.append({
                "text": obj.properties.get("text"),
                "metadata": {
                    "source": obj.properties.get("source"),
                    "filename": obj.properties.get("filename"),
                    "page": obj.properties.get("page"),
                    "title": obj.properties.get("title"),
                    "chunk_id": obj.properties.get("chunk_id"),
                },
                "score": obj.metadata.score if obj.metadata else None
            })
            
        return results

    def close(self) -> None:
        """
        Close the client connection cleanly.
        """
        self.client.close()
