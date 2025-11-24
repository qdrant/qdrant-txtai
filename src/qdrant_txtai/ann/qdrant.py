import warnings
from txtai.ann import ANN

from grpc import RpcError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
        PointIdsList,
        VectorParams,
        Distance,
        SearchRequest,
        SearchParams,
    )



class Qdrant(ANN):
    """
    ANN implementation using Qdrant - https://qdrant.tech as a backend.
    """

    DISTANCE_MAPPING = {
        "cosine": Distance.COSINE,
        "l2": Distance.EUCLID,
        "ip": Distance.DOT,
        "l1": Distance.MANHATTAN,
    }

    def __init__(self, config):
        super().__init__(config)

        self.qdrant_config = self.config.get("qdrant", {})
        self.collection_name = self.qdrant_config.get("collection", "txtai-embeddings")
        self.qdrant_client = QdrantClient(
            location=self.qdrant_config.get("location"),
            url=self.qdrant_config.get("url"),
            port=self.qdrant_config.get("port", 6333),
            grpc_port=self.qdrant_config.get("grpc_port", 6334),
            prefer_grpc=self.qdrant_config.get("prefer_grpc", False),
            https=self.qdrant_config.get("https"),
            api_key=self.qdrant_config.get("api_key"),
            prefix=self.qdrant_config.get("prefix"),
            timeout=self.qdrant_config.get("timeout"),
            host=self.qdrant_config.get("host"),
            path=self.qdrant_config.get("path"),
            grpc_options=self.qdrant_config.get("grpc_options"),
            check_compatibility=False,  # Disable version check warning
        )

        # Initial offset is set to the number of existing rows
        try:
            self.config["offset"] = self.count()
        except (UnexpectedResponse, RpcError, ValueError):
            self.config["offset"] = 0

    def index(self, embeddings):
        vector_size = self.config.get("dimensions")
        metric_name = self.config.get("metric", "cosine")
        if metric_name not in self.DISTANCE_MAPPING:
            raise ValueError(f"Unsupported Qdrant similarity metric: {metric_name}")
        collection_config = self.qdrant_config.get("collection_config", {})

        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=self.DISTANCE_MAPPING[metric_name],
            ),
            **collection_config,
        )

        self.config["offset"] = 0
        self.append(embeddings)

    def append(self, embeddings):
        offset = self.config.get("offset", 0)
        new_count = embeddings.shape[0]
        ids = list(range(offset, offset + new_count))

        # Use upsert instead of deprecated upload_collection
        # Convert numpy array to list of lists for compatibility
        vectors = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

        # Create points for upsert
        points = [
            {
                "id": idx,
                "vector": vector
            }
            for idx, vector in zip(ids, vectors)
        ]

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        self.config["offset"] += new_count

    def delete(self, ids):
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def search(self, queries, limit):
        search_params = self.qdrant_config.get("search_params", {})

        # Handle batch search using query_points
        # since search_batch has been deprecated in newer qdrant-client versions
        results = []
        for query in queries:
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query.tolist(),
                limit=limit,
                search_params=SearchParams(**search_params) if search_params else None,
            )
            results.append([(point.id, point.score) for point in search_result.points])

        return results

    def count(self):
        result = self.qdrant_client.count(
            collection_name=self.collection_name,
        )
        return result.count

    def load(self, path):
        warnings.warn(
            "Trying to call .load method on Qdrant ANN backend. " "This is redundant and won't have any effect.",
            UserWarning,
        )

    def save(self, path):
        warnings.warn(
            "Trying to call .save method on Qdrant ANN backend. " "This is redundant and won't have any effect.",
            UserWarning,
        )