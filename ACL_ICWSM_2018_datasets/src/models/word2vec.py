import pandas as pd
import numpy as np
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim

class word2vec:
	def __init__(self, word2vec_path):
		self.EMBEDDING_DIM = 300
		self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
		# self.word2vec = []

	# Pad data.
	def pad(self, data):
		self.tokenizer = Tokenizer()
		self.tokenizer.fit_on_texts(data)
		sequenced_data = self.tokenizer.texts_to_sequences(data)

		self.MAX_SEQUENCE_LENGTH = 0
		for tokens in sequenced_data:
			self.MAX_SEQUENCE_LENGTH = max(self.MAX_SEQUENCE_LENGTH, len(tokens))

		return list(pad_sequences(sequenced_data, maxlen=self.MAX_SEQUENCE_LENGTH))

	# Construct word2vec embeddings.
	def create_embeddings(self, data):
		word_index = self.tokenizer.word_index
		self.embedding_weights = np.zeros((len(word_index) + 1, self.EMBEDDING_DIM))

		for word, index in word_index.items():
			self.embedding_weights[index, :] = self.word2vec[word] if word in self.word2vec else np.random.rand(self.EMBEDDING_DIM)

	# Compute average embedding weights for each data point.
	def compute_average_weights(self, data):
		averages = [
			np.average([self.embedding_weights[token] for token in tokens], axis=0)
			for tokens in data
		]

		return averages

	def compute_all_weights(self, data):
		weights = np.zeros(shape=(len(data), self.MAX_SEQUENCE_LENGTH, self.EMBEDDING_DIM))
		for i, tokens in enumerate(data):
			for j, token in enumerate(tokens):
				weights[i, j, :] = self.embedding_weights[token]

		return weights
