import pandas as pd
import json
import csv
import sys

def extract_id(text):
    return text.split(',')[1].split(':')[1]

df_train = pd.read_csv('../../original_data/nepal/2015_Nepal_Earthquake_train.tsv', sep='\t', encoding='iso-8859-1')
df_train = df_train.apply(lambda x: json.dumps({'tweet_id': str(x['tweet_id']), 'text': x['text'], 'label': x['label']}), axis=1)

df_dev = pd.read_csv('../../original_data/nepal/2015_Nepal_Earthquake_dev.tsv', sep='\t', encoding='iso-8859-1')
df_dev = df_dev.apply(lambda x: json.dumps({'tweet_id': str(x['tweet_id']), 'text': x['text'], 'label': x['label']}), axis=1)

df_test = pd.read_csv('../../original_data/nepal/2015_Nepal_Earthquake_test.tsv', sep='\t', encoding='iso-8859-1')
df_test = df_test.apply(lambda x: json.dumps({'tweet_id': str(x['tweet_id']), 'text': x['text'], 'label': x['label']}), axis=1)

with open('../../downloaded_tweets/ids_all.json', 'w') as outputfile:
    for index, item in df_train.iteritems():
        outputfile.write(item)
        outputfile.write('\n')

    for index, item in df_dev.iteritems():
        outputfile.write(item)
        outputfile.write('\n')

    for index, item in df_test.iteritems():
        outputfile.write(item)
        outputfile.write('\n')
