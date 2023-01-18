from txtai.app import Application
from txtai.embeddings import Embeddings

config = Application.read("./app-cloud.yml")
embeddings = Embeddings(config["embeddings"])
embeddings.index([(0, "Correct", None), (1, "Not what we hoped", None)])
result = embeddings.search("positive", 1)
print(result)
