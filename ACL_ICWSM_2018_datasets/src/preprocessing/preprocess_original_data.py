import pandas as pd
import json
import csv
import sys

input_file_base_name = sys.argv[1]
output_file = sys.argv[2]

df_train = pd.read_csv(input_file_base_name + '_train.tsv', sep='\t', encoding='iso-8859-1', quoting=csv.QUOTE_NONE)
df_train = df_train.apply(lambda x: json.dumps({'tweet_id': str(x['tweet_id']), 'text': x['text'], 'label': x['label']}), axis=1)

df_dev = pd.read_csv(input_file_base_name + '_dev.tsv', sep='\t', encoding='iso-8859-1', quoting=csv.QUOTE_NONE)
df_dev = df_dev.apply(lambda x: json.dumps({'tweet_id': str(x['tweet_id']), 'text': x['text'], 'label': x['label']}), axis=1)

df_test = pd.read_csv(input_file_base_name + '_test.tsv', sep='\t', encoding='iso-8859-1', quoting=csv.QUOTE_NONE)
df_test = df_test.apply(lambda x: json.dumps({'tweet_id': str(x['tweet_id']), 'text': x['text'], 'label': x['label']}), axis=1)

with open(output_file, 'w') as of:
    for index, item in df_train.iteritems():
        of.write(item)
        of.write('\n')

    for index, item in df_dev.iteritems():
        of.write(item)
        of.write('\n')

    for index, item in df_test.iteritems():
        of.write(item)
        of.write('\n')
