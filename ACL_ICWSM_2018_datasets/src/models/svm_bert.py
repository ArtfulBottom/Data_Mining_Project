import pandas as pd
import numpy as np
import sys

from sklearn.svm import SVC
from sklearn.preprocessing import normalize


module_bertLoader = __import__('bert_loader')

# Read input data.
inputfile = sys.argv[1]
bert_path = sys.argv[2]

df = pd.read_csv(inputfile)
sampleIds  = df.sample(frac=0.2, random_state=25).index
test_data  = df[df.index.isin(sampleIds)]
train_data = df.drop(df.index[sampleIds])

# Obtain average vectors for train and test data.
embeddings_bert        = module_bertLoader.loadBertEmb(bert_path, 4)
embeddings_bert_test   = embeddings_bert[embeddings_bert.index.isin(sampleIds)]
embeddings_bert_train  = embeddings_bert.drop(embeddings_bert.index[sampleIds])

class_column = 'relevance_label'

# Evaluate SVM without temporal dimension.
svm_model = SVC(kernel='linear', C=100).fit(embeddings_bert_train, train_data[class_column])

train_accuracy = svm_model.score(embeddings_bert_train, train_data[class_column])
test_accuracy  = svm_model.score(embeddings_bert_test, test_data[class_column])

print('SVM + bert train accuracy: %.4f' % (train_accuracy * 100))
print('SVM + bert test_accuracy: %.4f' % (test_accuracy * 100))

# Evaluate SVM with only temporal dimension.
svm_model.fit([[label] for label in train_data['day_label']], train_data[class_column])

train_accuracy = svm_model.score([[label] for label in train_data['day_label']], train_data[class_column])
test_accuracy  = svm_model.score([[label] for label in test_data['day_label']], test_data[class_column])

print('SVM + time train accuracy: %.4f' % (train_accuracy * 100))
print('SVM + time test_accuracy: %.4f' % (test_accuracy * 100))

# Add temporal dimension to features.
average_weights_train_temporal = normalize(np.reshape(np.asarray(train_data['day_label']), (len(train_data['day_label']), 1)))
average_weights_test_temporal  = normalize(np.reshape(np.asarray(test_data['day_label']), (len(test_data['day_label']), 1)))

average_weights_train = np.concatenate((embeddings_bert_train, average_weights_train_temporal), axis=1)
average_weights_test  = np.concatenate((embeddings_bert_test, average_weights_test_temporal), axis=1)

# Evaluate SVM with w2v + temporal dimension.
svm_model.fit(average_weights_train, train_data[class_column])

train_accuracy = svm_model.score(average_weights_train, train_data[class_column])
test_accuracy  = svm_model.score(average_weights_test, test_data[class_column])

print('SVM + bert + time train accuracy: %.4f' % (train_accuracy * 100))
print('SVM + bert + time test_accuracy: %.4f' % (test_accuracy * 100))
