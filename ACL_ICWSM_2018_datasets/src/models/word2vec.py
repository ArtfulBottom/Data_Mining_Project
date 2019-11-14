import pandas as pd
import numpy as np
import sys
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import gensim
import tensorflow as tf

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
		
		self.embedding_layer = Embedding(self.embedding_weights.shape[0], self.embedding_weights.shape[1], 
			input_length=self.MAX_SEQUENCE_LENGTH, weights=[self.embedding_weights], trainable=False)

	# Compute average embedding weights for each data point.
	def compute_average_weights(self, data):
		# init_g = tf.compat.v1.global_variables_initializer()
		# init_l = tf.compat.v1.local_variables_initializer()

		# with tf.Session() as sess:
		# 	sess.run(init_g)
		# 	sess.run(init_l)

		# 	tensor = tf.convert_to_tensor(np.asarray(list(data), np.float32), dtype=tf.float32)
		# 	embedded = self.embedding_layer(tensor)
		# 	print(embedded.eval())
		# 	print(' ')
			
		# 	return np.average(embedded.eval(), axis=1)
		averages = [
			np.average([self.embedding_weights[token] for token in tokens], axis=0)
			for tokens in data
		]

		return averages
