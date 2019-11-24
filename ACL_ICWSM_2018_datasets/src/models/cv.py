import pandas as pd
import numpy as np
import sys
from word2vec import *
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

def create_w2v_bilstm(learn_rate=0.001, batch_size=64, dropout=0.4):
	global w2v

	bi_lstm = Sequential()
	bi_lstm.add(Dense(batch_size, input_shape=(w2v.MAX_SEQUENCE_LENGTH, w2v.EMBEDDING_DIM)))
	bi_lstm.add(Bidirectional(LSTM(w2v.EMBEDDING_DIM, dropout=dropout, activation='relu')))
	bi_lstm.add(Dense(1, activation='sigmoid'))

	optimizer = Adam(lr=learn_rate)
	bi_lstm.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

	return bi_lstm

def create_w2v_time(learn_rate=0.001, batch_size=64, dropout=0.4):
	global w2v

	nn = Sequential()
	nn.add(Dense(batch_size, input_dim=1, activation='relu'))
	nn.add(Dropout(dropout))
	nn.add(Dense(1, activation='sigmoid'))

	optimizer = Adam(lr=learn_rate)
	nn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

	return nn

def create_w2v_bilstm_time(learn_rate=0.001, batch_size=64, dropout=0.4):
	global w2v

	bi_lstm = Sequential()
	bi_lstm.add(Dense(batch_size, input_shape=(w2v.MAX_SEQUENCE_LENGTH, w2v.EMBEDDING_DIM)))
	bi_lstm.add(Bidirectional(LSTM(w2v.EMBEDDING_DIM, dropout=dropout, activation='relu')))
	bi_lstm.add(Dense(batch_size, activation='relu'))

	nn = Sequential()
	nn.add(Dense(batch_size, input_dim=1, activation='relu'))
	nn.add(Dropout(dropout))

	merged_out = Add()([bi_lstm.output, nn.output])
	merged_out = Dense(1, activation='sigmoid')(merged_out)

	model = Model([bi_lstm.input, nn.input], merged_out)
	optimizer = Adam(lr=learn_rate)
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

	return model

def kfold_cv(data, word2vec_path, model_choice):
	# Setup data: w2v, time, and w2v + time.
	global w2v 
	w2v = word2vec(word2vec_path)
	data['tokens'] = w2v.pad(data['text'].tolist())
	w2v.create_embeddings(data['tokens'])

	average_weights = w2v.compute_average_weights(data['tokens'])
	time_feature = [[label] for label in data['day_label']]

	time_labels = np.reshape(np.asarray(data['day_label']), (len(data['day_label']), 1))
	average_weights_time = np.concatenate((average_weights, time_labels), axis=1)

	class_label = 'relevance_label'

	# Perform grid search CV.
	if model_choice == 'LR':
		parameters = {'C': (0.1, 1, 100, 1000)}
		gs = GridSearchCV(LogisticRegression(max_iter=1000), parameters, cv=5)

		# Find best parameters for w2v.
		gs.fit(average_weights, data[class_label])
		print('Best parameters for LR word2vec: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.best_score_ * 100), end='\n\n')

		# Find best parameters for time.
		gs.fit(time_feature, data[class_label])
		print('Best parameters for LR time: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.best_score_ * 100), end='\n\n')

		# Find best parameters for w2v + time.
		gs.fit(average_weights_time, data[class_label])
		print('Best parameters for LR word2vec + time: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.best_score_ * 100))

	elif model_choice == 'SVM':
		average_weights = normalize(average_weights, axis=0)
		average_weights_time = normalize(average_weights_time, axis=0)

		parameters = {'kernel': ('linear', 'rbf'), 'C': (0.1, 1, 100, 1000)}
		gs = GridSearchCV(SVC(), parameters, cv=5)

		# Find best parameters for w2v.
		gs.fit(average_weights, data[class_label])
		print('Best parameters for SVM word2vec: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.best_score_ * 100), end='\n\n')

		# Find best parameters for time.
		gs.fit(time_feature, data[class_label])
		print('Best parameters for SVM time: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.best_score_ * 100), end='\n\n')

		# Find best parameters for w2v + time.
		gs.fit(average_weights_time, data[class_label])
		print('Best parameters for SVM word2vec + time: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.best_score_ * 100))

	elif model_choice == 'LSTM':
		batch_size = [64, 128]
		epochs = [4, 10]
		learn_rate = [0.001, 0.01]
		dropout = [0.2, 0.4]

		dev = data.sample(frac=0.2, random_state=37)
		data = data.drop(dev.index)

		param_grid = dict(learn_rate=learn_rate, batch_size=batch_size, epochs=epochs, dropout=dropout)
		weights = w2v.compute_all_weights(data['tokens'])

		best_score = 0
		for p in ParameterGrid(param_grid):
			w2v_bilstm_model = create_w2v_bilstm(learn_rate=p['learn_rate'], batch_size=p['batch_size'], dropout=p['dropout'])
			w2v_bilstm_model.fit(weights, data[class_label], batch_size=p['batch_size'], epochs=p['epochs'], verbose=0)

			accuracy = w2v_bilstm_model.evaluate(w2v.compute_all_weights(dev['tokens']), dev[class_label], batch_size=p['batch_size'], epochs=p['epochs'], verbose=0)[1]
			if accuracy > best_score:
				best_score = accuracy
				best_grid = p

		print('Best parameters for BiLSTM word2vec: %r' % best_grid)
		print('Corresponding accuracy: %.4f' % (best_score * 100))

		K.clear_session()

		best_score = 0
		for p in ParameterGrid(param_grid):
			w2v_time_model = create_w2v_time(learn_rate=p['learn_rate'], batch_size=p['batch_size'], dropout=p['dropout'])
			w2v_time_model.fit(np.asarray(data['day_label']), data[class_label], batch_size=p['batch_size'], epochs=p['epochs'], verbose=0)

			accuracy = w2v_time_model.evaluate(np.asarray(dev['day_label']), dev[class_label], batch_size=p['batch_size'], epochs=p['epochs'], verbose=0)[1]
			if accuracy > best_score:
				best_score = accuracy
				best_grid = p

		print('Best parameters for BiLSTM time: %r' % best_grid)
		print('Corresponding accuracy: %.4f' % (best_score * 100))

		K.clear_session()

		best_score = 0
		for p in ParameterGrid(param_grid):
			w2v_bilstm_time_model = create_w2v_bilstm_time(learn_rate=p['learn_rate'], batch_size=p['batch_size'], dropout=p['dropout'])
			w2v_bilstm_time_model.fit([weights, data['day_label']], data[class_label], batch_size=p['batch_size'], epochs=p['epochs'], verbose=0)

			accuracy = w2v_bilstm_time_model.evaluate([w2v.compute_all_weights(dev['tokens']), dev['day_label']], dev[class_label], batch_size=p['batch_size'], epochs=p['epochs'], verbose=0)[1]
			if accuracy > best_score:
				best_score = accuracy
				best_grid = p

		print('Best parameters for BiLSTM word2vec + time: %r' % best_grid)
		print('Corresponding accuracy: %.4f' % (best_score * 100))

		K.clear_session()

if __name__=='__main__':
	input_file = sys.argv[1]
	word2vec_path = sys.argv[2]
	model_choice = sys.argv[3]

	data = pd.read_csv(input_file)
	data = data.drop(data.sample(frac=0.2, random_state=25).index)

	kfold_cv(data, word2vec_path, model_choice)
