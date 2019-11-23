import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression

module_bertLoader = __import__('bert_loader')

	
# Read input data.
inputfile   = sys.argv[1]
bert_path   = sys.argv[2]
numOfLayers = int(sys.argv[3])
    
df         = pd.read_csv(inputfile)
sampleIds  = df.sample(frac=0.2, random_state=25).index
test_data  = df[df.index.isin(sampleIds)]
train_data = df.drop(df.index[sampleIds])

# Obtain average vectors for train and test data.
embeddings_bert        = module_bertLoader.loadBertEmb(bert_path, numOfLayers)
embeddings_bert_test   = embeddings_bert[embeddings_bert.index.isin(sampleIds)]
embeddings_bert_train  = embeddings_bert.drop(embeddings_bert.index[sampleIds])

class_column = 'relevance_label'

# Evaluate LR without temporal dimension.
lr_model = LogisticRegression(penalty='l2', max_iter=1000).fit(embeddings_bert_train, train_data[class_column])

train_accuracy = lr_model.score(embeddings_bert_train, train_data[class_column])
test_accuracy = lr_model.score(embeddings_bert_test, test_data[class_column])

print('LR + bert train accuracy: %.4f' % (train_accuracy * 100))
print('LR + bert test_accuracy: %.4f' % (test_accuracy * 100))

# Evaluate SVM with only temporal dimension.
lr_model.fit([[label] for label in train_data['day_label']], train_data[class_column])

train_accuracy = lr_model.score([[label] for label in train_data['day_label']], train_data[class_column])
test_accuracy = lr_model.score([[label] for label in test_data['day_label']], test_data[class_column])

print('LR + time train accuracy: %.4f' % (train_accuracy * 100))
print('LR + time test_accuracy: %.4f' % (test_accuracy * 100))

# Add temporal dimension to features.
train_time_labels = np.reshape(np.asarray(train_data['day_label']), (len(train_data['day_label']), 1))
test_time_labels = np.reshape(np.asarray(test_data['day_label']), (len(test_data['day_label']), 1))

embeddings_bert_train = np.concatenate((embeddings_bert_train, train_time_labels), axis=1)
embeddings_bert_test = np.concatenate((embeddings_bert_test, test_time_labels), axis=1)

# Evaluate LR with embeddings_bert + temporal dimension.
lr_model.fit(embeddings_bert_train, train_data[class_column])

train_accuracy = lr_model.score(embeddings_bert_train, train_data[class_column])
test_accuracy = lr_model.score(embeddings_bert_test, test_data[class_column])

print('LR + bert + time train accuracy: %.4f' % (train_accuracy * 100))
print('LR + bert + time test_accuracy: %.4f' % (test_accuracy * 100))
