import pandas as pd
import numpy as np
import sys
from word2vec import *
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import bert_loader as module_bertLoader

def kfold_cv(data, bert_path, model_choice):
	# Setup data: w2v, time, and w2v + time.
	time_feature = [[label] for label in data['day_label']]
	time_labels = np.reshape(np.asarray(data['day_label']), (len(data['day_label']), 1))
	average_weights_time = np.concatenate((average_weights, time_labels), axis=1)

    sampleIds  = df.sample(frac=0.2, random_state=25).index
    test_data  = df[df.index.isin(sampleIds)]
    train_data = df.drop(df.index[sampleIds])

    # Obtain average vectors for train and test data.
    # module_bertLoader      = __import__('bert_loader')
    embeddings_bert        = module_bertLoader.loadBertEmb(bert_path)
    embeddings_bert_test   = embeddings_bert[embeddings_bert.index.isin(sampleIds)]
    embeddings_bert_train  = embeddings_bert.drop(embeddings_bert.index[sampleIds])

	class_label = 'relevance_label'

	# Perform grid search CV.
	if model_choice == 'LR':
		parameters = {'C': (0.1, 1, 100, 1000)}
		gs = GridSearchCV(LogisticRegression(max_iter=1000), parameters, cv=5)

		# Find best parameters for w2v.
		gs.fit(average_weights, data[class_label])
		print('Best parameters for for LR bert: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.cv_results_['mean_test_score'] * 100), end='\n\n')

		# Find best parameters for time.
		gs.fit(time_feature, data[class_label])
		print('Best parameters for for LR time: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.cv_results_['mean_test_score'] * 100), end='\n\n')

		# Find best parameters for w2v + time.
		gs.fit(average_weights_time, data[class_label])
		print('Best parameters for for LR bert + time: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.cv_results_['mean_test_score'] * 100))

	elif model_choice == 'SVM':
		average_weights = normalize(average_weights, axis=0)
		average_weights_time = normalize(average_weights_time, axis=0)

		parameters = {'kernel': ('linear', 'rbf'), 'C': (0.1, 1, 100, 1000)}
		gs = GridSearchCV(SVC(), parameters, cv=5)

		# Find best parameters for w2v.
		gs.fit(average_weights, data[class_label])
		print('Best parameters for for SVM bert: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.cv_results_['mean_test_score'] * 100), end='\n\n')

		# Find best parameters for time.
		gs.fit(time_feature, data[class_label])
		print('Best parameters for for SVM time: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.cv_results_['mean_test_score'] * 100), end='\n\n')

		# Find best parameters for w2v + time.
		gs.fit(average_weights_time, data[class_label])
		print('Best parameters for for SVM bert + time: %r' % gs.best_params_)
		print('Corresponding accuracy: %.4f' % (gs.cv_results_['mean_test_score'] * 100))

if __name__=='__main__':
	input_file = sys.argv[1]
	bert_path = sys.argv[2]
	model_choice = sys.argv[3]

	data = pd.read_csv(input_file)

	kfold_cv(data, bert_path, model_choice)
