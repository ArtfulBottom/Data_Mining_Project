import pandas as pd
import numpy as np
import sys
from word2vec import *
from keras.models import *
from keras.layers import *
from keras import backend as K

if __name__=='__main__':	
	# Read input data.
	input_file = sys.argv[1]
	word2vec_path = sys.argv[2]

	df = pd.read_csv(input_file)
	w2v = word2vec(word2vec_path)

	df['tokens'] = w2v.pad(df['text'].tolist())
	w2v.create_embeddings(df['tokens'])

	test_data = df.sample(frac=0.2, random_state=25)
	train_data = df.drop(test_data.index)
	class_column = 'relevance_label'

	# Obtain average vectors for train and test data.
	average_weights_train = w2v.compute_all_weights(train_data['tokens'])
	average_weights_test = w2v.compute_all_weights(test_data['tokens'])

	# Setup hyperparameters.
	batch_size = 32
	dropout = 0.5
	num_epochs = 4

	# Evaluate NN without temporal dimension.
	bi_lstm = Sequential()
	bi_lstm.add(Dense(batch_size, input_shape=(w2v.MAX_SEQUENCE_LENGTH, w2v.EMBEDDING_DIM)))
	bi_lstm.add(Bidirectional(LSTM(w2v.EMBEDDING_DIM, dropout=0.5, recurrent_dropout=0.5, activation='relu')))
	bi_lstm.add(Dense(1, activation='sigmoid'))

	bi_lstm.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
	bi_lstm.fit(average_weights_train, np.asarray(train_data[class_column]), batch_size=batch_size, epochs=num_epochs)

	print('BiLSTM + word2vec train accuracy: %.4f' % (bi_lstm.evaluate(average_weights_train, train_data[class_column], batch_size=batch_size, epochs=num_epochs)[1] * 100))
	print('BiLSTM + word2vec test accuracy: %.4f' % (bi_lstm.evaluate(average_weights_test, test_data[class_column], batch_size=batch_size, epochs=num_epochs)[1] * 100))

	K.clear_session()

	# Evaluate NN with only temporal dimension.
	nn = Sequential()
	nn.add(Dense(batch_size, input_dim=1, activation='relu'))
	nn.add(Dropout(0.5))
	nn.add(Dense(1, activation='sigmoid'))

	nn.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
	nn.fit(np.asarray(train_data['day_label']), np.asarray(train_data[class_column]), batch_size=batch_size, epochs=num_epochs)
	print('NN_time train accuracy: %.4f' % (nn.evaluate(np.asarray(train_data['day_label']), train_data[class_column], batch_size=batch_size, epochs=num_epochs)[1] * 100))
	print('NN_time test accuracy: %.4f' % (nn.evaluate(np.asarray(test_data['day_label']), test_data[class_column], batch_size=batch_size, epochs=num_epochs)[1] * 100))

	K.clear_session()

	# Evaluate NN with w2v + temporal dimension.
	bi_lstm = Sequential()
	bi_lstm.add(Dense(batch_size, input_shape=(w2v.MAX_SEQUENCE_LENGTH, w2v.EMBEDDING_DIM)))
	bi_lstm.add(Bidirectional(LSTM(w2v.EMBEDDING_DIM, dropout=0.5, recurrent_dropout=0.5, activation='relu')))
	bi_lstm.add(Dense(32, activation='relu'))

	nn = Sequential()
	nn.add(Dense(batch_size, input_dim=1, activation='relu'))
	nn.add(Dropout(0.5))

	merged_out = Add()([bi_lstm.output, nn.output])
	merged_out = Dense(1, activation='sigmoid')(merged_out)

	model = Model([bi_lstm.input, nn.input], merged_out)
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

	model.fit([average_weights_train, np.asarray(train_data['day_label'])], np.asarray(train_data[class_column]), batch_size=batch_size, epochs=num_epochs)
	print('BiLSTM + NN_time train accuracy: %.4f' % (model.evaluate([average_weights_train, np.asarray(train_data['day_label'])], train_data[class_column], batch_size=batch_size, epochs=num_epochs)[1] * 100))
	print('BiLSTM + NN_time test accuracy: %.4f' % (model.evaluate([average_weights_test, np.asarray(test_data['day_label'])], test_data[class_column], batch_size=batch_size, epochs=num_epochs)[1] * 100))

	K.clear_session()
