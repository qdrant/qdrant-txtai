import txtai
import os

os.environ["config"] = "app.yml"

embeddings = txtai.Embeddings()
embeddings.index(["Correct", "Not what we hoped"])
result = embeddings.search("positive", 1)
print(result)
