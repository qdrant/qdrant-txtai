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
docker run -p 6333:6333 -p:6334:6334 qdrant/qdrant:v1.0.1
```

Running the txtai application might be done either programmatically or by 
providing configuration in a YAML file.

### Programmatically

```python
from txtai.embeddings import Embeddings

embeddings = Embeddings({
    "path": "sentence-transformers/all-MiniLM-L6-v2",
    "backend": "qdrant_txtai.ann.qdrant.Qdrant",
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

The example below presents all the available options, if we connect to Qdrant server:

```yaml
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  backend: qdrant_txtai.ann.qdrant.Qdrant
  metric: l2 # allowed values: l2 / cosine / ip
  qdrant:
    url: qdrant.host
    port: 6333
    grpc_port: 6334
    prefer_grpc: true
    collection: CustomCollectionName
    https: true # for Qdrant Cloud
    api_key: XYZ # for Qdrant Cloud
    hnsw:
      m: 8
      ef_construct: 256
      full_scan_threshold:
      ef_search: 512
```

### Local in-memory/disk-persisted mode

Qdrant Python client, from version 1.1.1, supports local in-memory/disk-persisted mode. 
That's a good choice for any test scenarios and quick experiments in which you do not 
plan to store lots of vectors. In such a case spinning a Docker container might be even 
not required.

#### In-memory storage

In case you want to have a transient storage, for example in case of automated tests 
launched during your CI/CD pipeline, using Qdrant Local mode with in-memory storage 
might be a preferred option.

```yaml
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  backend: qdrant_txtai.ann.qdrant.Qdrant
  metric: l2 # allowed values: l2 / cosine / ip
  qdrant:
    location: ':memory:'
    prefer_grpc: true
```

#### On disk storage

However, if you prefer to keep the vectors between different runs of your application, 
it might be better to use on disk storage and pass the path that should be used to 
persist the data.

```yaml
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  backend: qdrant_txtai.ann.qdrant.Qdrant
  metric: l2 # allowed values: l2 / cosine / ip
  qdrant:
    path: '/home/qdrant/storage_local'
    prefer_grpc: true
```
