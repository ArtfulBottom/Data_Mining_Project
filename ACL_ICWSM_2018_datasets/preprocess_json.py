import pandas as pd
import numpy as np
import csv
import json

dic = {}

with open('./downloaded_tweets/ids.json.data.json') as fr:
	for l in fr:
		jobj = json.loads(l.strip())
		dic[jobj['id_str']] = {'created_at': jobj['created_at']}

with open('./downloaded_tweets/ids.json.label.json') as fr:
	for l in fr:
		jobj = json.loads(l.strip())
		if dic.get(jobj['tweet_id']) is not None:
			dic[jobj['tweet_id']]['label'] = jobj['label']

relevant_dates_dic = {}
irrelevant_dates_dic = {}
for key, value in dic.items():
	tokens = value['created_at'].split()
	dates_key = tokens[1] + tokens[2]

	if value['label'] == 'relevant':
		if relevant_dates_dic.get(dates_key) is None:
			relevant_dates_dic[dates_key] = 0
		relevant_dates_dic[dates_key] += 1
	else:
		if irrelevant_dates_dic.get(dates_key) is None:
			irrelevant_dates_dic[dates_key] = 0
		irrelevant_dates_dic[dates_key] += 1

print(relevant_dates_dic)
print(irrelevant_dates_dic)

df = pd.DataFrame.from_dict(dic, orient='index')
print(df.head())

