import warnings

import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointIdsList,
    VectorParams,
    Distance,
    HnswConfigDiff,
    SearchRequest, SearchParams,
)
from txtai.ann import ANN


class Qdrant(ANN):
    """
    ANN implementation using Qdrant server as a backend.
    """

    DISTANCE = {
        "cosine": Distance.COSINE,
        "l2": Distance.EUCLID,
        "ip": Distance.DOT,
    }

    def __init__(self, config):
        super().__init__(config)

        # Load Qdrant specific configuration from the nested configuration dict
        self.qdrant_config = self.config.get("qdrant", {})
        hostname = self.qdrant_config.get("host", "localhost")
        port = self.qdrant_config.get("port", 6333)
        grpc_port = self.qdrant_config.get("grpc_port", 6334)
        prefer_grpc = self.qdrant_config.get("prefer_grpc", False)
        self.qdrant_client = QdrantClient(hostname, port, grpc_port, prefer_grpc)
        self.collection_name = self.qdrant_config.get("collection", "embeddings")

        # Initial offset is set to the number of existing rows
        try:
            self.config["offset"] = self.count()
        except qdrant_client.http.exceptions.UnexpectedResponse:
            self.config["offset"] = 0

    def load(self, path):
        # Since Qdrant does not rely on files, there is no need to load anything
        # from given path. Instead, the file path is used as a collection name,
        # effectively allowing to use different embeddings.
        warnings.warn(
            "Trying to call .load method on Qdrant ANN backend. "
            "It won't have any effect.", UserWarning,
        )

    def index(self, embeddings):
        vector_size = self.config.get("dimensions")
        metric_name = self.config.get("metric", "cosine")
        hnsw_config = self.qdrant_config.get("hnsw", {})

        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=self.DISTANCE[metric_name],
            ),
            hnsw_config=HnswConfigDiff(
                m=hnsw_config.get("m"),
                ef_construct=hnsw_config.get("ef_construct"),
                full_scan_threshold=hnsw_config.get("full_scan_threshold"),
            ),
        )

        self.config["offset"] = 0
        self.append(embeddings)

    def append(self, embeddings):
        offset = self.config.get("offset", 0)
        new = embeddings.shape[0]
        ids = list(range(offset, offset + new))
        self.qdrant_client.upload_collection(
            collection_name=self.collection_name,
            vectors=embeddings,
            ids=ids,
        )
        self.config["offset"] += new

    def delete(self, ids):
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def search(self, queries, limit):
        hnsw_config = self.qdrant_config.get("hnsw", {})
        ef_search = hnsw_config.get("ef_search")
        search_results = self.qdrant_client.search_batch(
            collection_name=self.collection_name,
            requests=[
                SearchRequest(
                    vector=query.tolist(),
                    params=SearchParams(hnsw_ef=ef_search),
                    limit=limit,
                )
                for query in queries
            ],
        )

        results = []
        for search_result in search_results:
            results.append([(entry.id, entry.score) for entry in search_result])
        return results

    def count(self):
        result = self.qdrant_client.count(
            collection_name=self.collection_name,
        )
        return result.count

    def save(self, path):
        # All the indexed embeddings are already saved in a Qdrant collection,
        # so there is no further need to perform any other action.
        pass
