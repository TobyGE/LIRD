import numpy as np

class Embeddings:
	def __init__(self, item_embeddings):
		self.item_embeddings = item_embeddings

	def size(self):
		return self.item_embeddings.shape[1]

	def get_embedding_vector(self):
		return self.item_embeddings

	def get_embedding(self, item_index):
		return self.item_embeddings[item_index]

	def embed(self, item_list):
		return np.array ([self.get_embedding (item) for item in item_list])