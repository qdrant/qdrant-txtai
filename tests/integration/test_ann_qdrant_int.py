from txtai.embeddings import Embeddings


def test_in_memory_embeddings():
    config = {
        "embeddings": {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "backend": "qdrant_txtai.ann.qdrant.Qdrant",
        "qdrant": {
            "location": ":memory:",
        },
    }

    embeddings = Embeddings(config)
    embeddings.index([(0, "Correct", None), (1, "Not what we hoped", None)])
    result = embeddings.search("positive", 1)

    assert result is not None
    assert len(result) == 1
    assert result[0][0] == 0
    assert result[0][1] > 0.0
