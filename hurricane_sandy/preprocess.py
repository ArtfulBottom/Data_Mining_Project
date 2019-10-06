import pandas as pd
import json
import csv

def extract_id(text):
    return text.split(',')[1].split(':')[1]

df = pd.read_csv('release.txt', sep='\t')
df = df.apply(lambda x: json.dumps({'tweet_id': extract_id(x['id']), 'date': x['date'], 'sandy_keyword': x['sandy_keyword']}), axis=1)
print(df.head())
df.to_csv('ids.json', header=False, index=False, quoting=csv.QUOTE_NONE, sep='|')
