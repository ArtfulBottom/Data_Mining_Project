import pandas as pd
import numpy as np
import sys
from word2vec import *
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

if __name__=='__main__':	
	# Read input data.
	inputfile = sys.argv[1]
	word2vec_path = sys.argv[2]

	df = pd.read_csv(inputfile)
	w2v = word2vec(word2vec_path)

	df['tokens'] = w2v.pad(df['text'].tolist())
	w2v.create_embeddings(df['tokens'])

	test_data = df.sample(frac=0.2, random_state=25)
	valid_data = df.drop(test_data.index).sample(frac=0.2, random_state=47)
	train_data = df.drop(test_data.index).drop(valid_data.index)
	class_column = 'relevance_label'

	# Obtain average vectors for train and test data.
	average_weights_train = normalize(w2v.compute_average_weights(train_data['tokens']), axis=0)
	average_weights_test = normalize(w2v.compute_average_weights(test_data['tokens']), axis=0)

	# Evaluate LR without temporal dimension.
	svm_model = SVC(kernel='linear', C=100).fit(average_weights_train, train_data[class_column])

	train_predictions = svm_model.predict(average_weights_train)
	test_predictions = svm_model.predict(average_weights_test)

	train_accuracy = np.average(train_predictions == train_data[class_column])
	test_accuracy = np.average(test_predictions == test_data[class_column])

	print('SVM + word2vec train accuracy: %.4f' % train_accuracy)
	print('SVM + word2vec test_accuracy: %.4f' % test_accuracy)

	# Add temporal dimension to features.
	average_weights_train_temporal = normalize(np.reshape(np.asarray(train_data['day_label']), (len(train_data['day_label']), 1)))
	average_weights_test_temporal = normalize(np.reshape(np.asarray(test_data['day_label']), (len(test_data['day_label']), 1)))

	average_weights_train = np.concatenate((average_weights_train, average_weights_train_temporal), axis=1)
	average_weights_test = np.concatenate((average_weights_test, average_weights_test_temporal), axis=1)

	# Evaluate LR with temporal dimension.
	svm_model.fit(average_weights_train, train_data[class_column])

	train_predictions = svm_model.predict(average_weights_train)
	test_predictions = svm_model.predict(average_weights_test)

	# # Evaluate LR with temporal dimension.
	# svm_model.fit([[label] for label in train_data['day_label']], train_data[class_column])

	# train_predictions = svm_model.predict([[label] for label in train_data['day_label']])
	# test_predictions = svm_model.predict([[label] for label in test_data['day_label']])

	train_accuracy = np.average(train_predictions == train_data[class_column])
	test_accuracy = np.average(test_predictions == test_data[class_column])

	print('SVM + word2vec + day train accuracy: %.4f' % train_accuracy)
	print('SVM + word2vec + day test_accuracy: %.4f' % test_accuracy)
