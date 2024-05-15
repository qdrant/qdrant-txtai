import os
import tempfile


import numpy as np

from txtai.ann import ANNFactory


class TestANN:
    """
    ANN tests.
    """

    def testQdrantInMemory(self):
        """
        Test Qdrant backend with in-memory storage.
        We run simpler tests here as Qdrant in-memory storage does not support concurrent access.
        """
        dimensions = 300
        data = np.random.rand(1, dimensions).astype(np.float32)

        ann = ANNFactory.create(
            {
                "backend": "qdrant_txtai.ann.qdrant.Qdrant",
                "qdrant": {"location": ":memory:"},
                "dimensions": dimensions,
            }
        )

        ann.index(data)

        assert ann.count() == 1

        ann.append(data)

        assert ann.count() == 2

        ann.delete([0])
        assert ann.count() == 1

    def testQdrantServer(self):
        """
        Test Qdrant backend with a server running at localhost:6333
        """

        self.runTests(
            "qdrant_txtai.ann.qdrant.Qdrant",
            {"qdrant": {"url": "http://localhost:6333"}},
        )

    def runTests(self, name, params=None, update=True):
        """
        Runs a series of standard backend tests.
        """
        assert self.backend(name, params).config["backend"], name
        assert self.save(name, params).count(), 10000

        assert self.append(name, params, 500).count(), 10500
        assert self.delete(name, params, [0, 1]).count(), 9998
        assert self.delete(name, params, [100000]).count(), 10000

        assert self.search(name, params) > 0

    def backend(self, name, params=None, length=10000):
        """
        Test the backend.
        """

        data = np.random.rand(length, 300).astype(np.float32)
        self.normalize(data)

        config = {"backend": name, "dimensions": data.shape[1]}
        if params:
            config.update(params)

        model = ANNFactory.create(config)
        model.index(data)

        return model

    def append(self, name, params=None, length=500):
        """
        Appends new data to index.
        """

        model = self.backend(name, params)

        data = np.random.rand(length, 300).astype(np.float32)
        self.normalize(data)

        model.append(data)

        return model

    def delete(self, name, params=None, ids=None):
        """
        Deletes data from index.
        """

        model = self.backend(name, params)
        model.delete(ids)

        return model

    def save(self, name, params=None):
        """
        Test save/load.
        """

        model = self.backend(name, params)

        index = os.path.join(tempfile.gettempdir(), "ann")

        model.save(index)
        model.load(index)

        return model

    def search(self, name, params=None):
        """
        Test ANN search.
        """

        model = self.backend(name, params)

        query = np.random.rand(300).astype(np.float32)
        self.normalize(query)

        return model.search(np.array([query]), 1)[0][0][1]

    def normalize(self, embeddings):
        """
        Normalizes embeddings using L2 normalization. Operation applied directly on array.
        """

        if len(embeddings.shape) > 1:
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        else:
            embeddings /= np.linalg.norm(embeddings)
