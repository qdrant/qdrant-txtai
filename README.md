# qdrant-txtai

[txtai](https://github.com/neuml/txtai) simplifies building AI-powered semantic 
search applications using Transformers. It leverages the neural embeddings and
their properties to encode high-dimensional data in a lower-dimensional space 
and allows to find similar objects based on their embeddings' proximity. 

Implementing such application in real-world use cases requires storing the
embeddings in an efficient way though, namely in a vector database like 
[Qdrant](https://qdrant.tech). It offers not only a powerful engine for neural
search, but also allows setting up a whole cluster if your data does not fit
a single machine anymore. It is production grade and can be launched easily
with Docker.

Combining the easiness of txtai with Qdrant's performance enables you to build
production-ready semantic search applications way faster than before.

## Installation

The library might be installed with pip as following:

```bash
pip install qdrant-txtai
```

## Usage

Running the txtai application with Qdrant as a vector storage requires launching
a Qdrant instance. That might be done easily with Docker:

```bash
docker run -p 6333:6333 -p:6334:6334 qdrant/qdrant:v0.10.2
```

Running the txtai application might be done either programmatically or by 
providing configuration in a YAML file.

### Programmatically

```python
from txtai.embeddings import Embeddings

embeddings = Embeddings({
    "embeddings": {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "backend": "qdrant_txtai.ann.qdrant.Qdrant",
    },
})
embeddings.index([(0, "Correct", None), (1, "Not what we hoped", None)])
result = embeddings.search("positive", 1)
print(result)
```

### Via YAML configuration

```yaml
# app.yml
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  backend: qdrant_txtai.ann.qdrant.Qdrant
```

```bash
CONFIG=app.yml uvicorn "txtai.api:app"
curl -X GET "http://localhost:8000/search?query=positive"
```

## Configuration properties

*qdrant-txtai* allows you to configure both the connection details, and some 
internal properties of the vector collection which may impact both speed and
accuracy. Please refer to [Qdrant docs](https://qdrant.github.io/qdrant/redoc/index.html#tag/collections/operation/create_collection)
if you are interested in the meaning of each property.

The example below presents all the available options:

```yaml
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  backend: qdrant_txtai.ann.qdrant.Qdrant
  metric: l2 # allowed values: l2 / cosine / ip
  qdrant:
    host: localhost
    port: 6333
    grpc_port: 6334
    prefer_grpc: true
    collection: CustomCollectionName
    hnsw:
      m: 8
      ef_construct: 256
      full_scan_threshold:
      ef_search: 512
```
