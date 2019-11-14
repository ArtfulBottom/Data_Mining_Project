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

	lr_model = LogisticRegression(penalty='l2', max_iter=500).fit(
		w2v.compute_average_weights(train_data['tokens']), train_data[class_column]
	)

	train_predictions = lr_model.predict(
		w2v.compute_average_weights(train_data['tokens'])
	)

	test_predictions = lr_model.predict(
		w2v.compute_average_weights(test_data['tokens'])
	)

	train_accuracy = np.average(train_predictions == train_data[class_column])
	test_accuracy = np.average(test_predictions == test_data[class_column])

	print(train_accuracy)
	print(test_accuracy)
