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
	valid_data = df.drop(test_data.index).sample(frac=0.2, random_state=47)
	train_data = df.drop(test_data.index).drop(valid_data.index)
	class_column = 'relevance_label'

	# Obtain average vectors for train and test data.
	average_weights_train = w2v.compute_average_weights(train_data['tokens'])
	average_weights_test = 	w2v.compute_average_weights(test_data['tokens'])

	# Evaluate LR without temporal dimension.
	lr_model = LogisticRegression(penalty='l2', max_iter=500).fit(average_weights_train, train_data[class_column])

	train_predictions = lr_model.predict(average_weights_train)
	test_predictions = lr_model.predict(average_weights_test)

	train_accuracy = np.average(train_predictions == train_data[class_column])
	test_accuracy = np.average(test_predictions == test_data[class_column])

	print('LR + word2vec train accuracy: %.4f' % train_accuracy)
	print('LR + word2vec test_accuracy: %.4f' % test_accuracy)

	# Add temporal dimension to features.
	average_weights_train_labels = np.reshape(np.asarray(train_data['day_label']), (len(train_data['day_label']), 1))
	average_weights_test_labels = np.reshape(np.asarray(test_data['day_label']), (len(test_data['day_label']), 1))

	average_weights_train = np.concatenate((average_weights_train, average_weights_train_labels), axis=1)
	average_weights_test = np.concatenate((average_weights_test, average_weights_test_labels), axis=1)

	# Evaluate LR with temporal dimension.
	lr_model.fit(average_weights_train, train_data[class_column])

	train_predictions = lr_model.predict(average_weights_train)
	test_predictions = lr_model.predict(average_weights_test)

	train_accuracy = np.average(train_predictions == train_data[class_column])
	test_accuracy = np.average(test_predictions == test_data[class_column])

	print('LR + word2vec + day train accuracy: %.4f' % train_accuracy)
	print('LR + word2vec + day test_accuracy: %.4f' % test_accuracy)
