import pandas as pd
import numpy as np
import csv
import json

data = pd.read_csv('../data/1_TrainingSet_2Class.csv')
# print(data.head())
temp = data[['Class', 'Id']].apply(lambda x: json.dumps({'tweet_id': x['Id'].strip('s'), 'relevance': x['Class']}), axis=1)
# print(data['Id'].head())
print(temp.head())
temp.to_csv('training.json', header=False, index=False, quoting=csv.QUOTE_NONE, sep='.')
# data['Id'].to_csv('training.json', header=False, index=False, quoting=csv.QUOTE_NONE)
# data['Id'].to_json('training.json')
