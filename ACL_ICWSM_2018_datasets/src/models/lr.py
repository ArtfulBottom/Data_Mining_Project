import pandas as pd
import numpy as np
import sys
from word2vec import *
from sklearn.linear_model import LogisticRegression

if __name__=='__main__':	
	# Read input data.
	inputfile = sys.argv[1]
	word2vec_path = sys.argv[2]

	df = pd.read_csv(inputfile)
	w2v = word2vec(word2vec_path)

	df['tokens'] = w2v.pad(df['text'].tolist())
	w2v.create_embeddings(df['tokens'])

	test_data = df.sample(frac=0.2, random_state=25)
	train_data = df.drop(test_data.index)
	class_column = 'relevance_label'

	# Obtain average vectors for train and test data.
	average_weights_train = w2v.compute_average_weights(train_data['tokens'])
	average_weights_test = 	w2v.compute_average_weights(test_data['tokens'])

	# Evaluate LR without temporal dimension.
	lr_model = LogisticRegression(penalty='l2', C=1000, max_iter=1000).fit(average_weights_train, train_data[class_column])

	train_accuracy = lr_model.score(average_weights_train, train_data[class_column])
	test_accuracy = lr_model.score(average_weights_test, test_data[class_column])

	print('LR + word2vec train accuracy: %.4f' % (train_accuracy * 100))
	print('LR + word2vec test_accuracy: %.4f' % (test_accuracy * 100))

	# Evaluate LR with only temporal dimension.
	lr_model = LogisticRegression(penalty='l2', C=0.1, max_iter=1000).fit([[label] for label in train_data['day_label']], train_data[class_column])

	train_accuracy = lr_model.score([[label] for label in train_data['day_label']], train_data[class_column])
	test_accuracy = lr_model.score([[label] for label in test_data['day_label']], test_data[class_column])

	print('LR + time train accuracy: %.4f' % (train_accuracy * 100))
	print('LR + time test_accuracy: %.4f' % (test_accuracy * 100))

	# Add temporal dimension to features.
	train_time_labels = np.reshape(np.asarray(train_data['day_label']), (len(train_data['day_label']), 1))
	test_time_labels = np.reshape(np.asarray(test_data['day_label']), (len(test_data['day_label']), 1))

	average_weights_train = np.concatenate((average_weights_train, train_time_labels), axis=1)
	average_weights_test = np.concatenate((average_weights_test, test_time_labels), axis=1)

	# Evaluate LR with w2v + temporal dimension.
	lr_model = LogisticRegression(penalty='l2', C=1, max_iter=1000).fit(average_weights_train, train_data[class_column])

	train_accuracy = lr_model.score(average_weights_train, train_data[class_column])
	test_accuracy = lr_model.score(average_weights_test, test_data[class_column])

	print('LR + word2vec + time train accuracy: %.4f' % (train_accuracy * 100))
	print('LR + word2vec + time test_accuracy: %.4f' % (test_accuracy * 100))
