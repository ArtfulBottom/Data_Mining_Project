import pandas as pd
import numpy as np
import csv
import json
import sys
from sklearn import preprocessing

input_base_file = sys.argv[1]
output_file = sys.argv[2]
dic = {}

# Process downloaded tweet IDs.
with open(input_base_file + '.data.json') as fr:
	for l in fr:
		jobj = json.loads(l.strip())
		dic[jobj['id_str']] = {'created_at': jobj['created_at']}

with open(input_base_file + '.label.json') as fr:
	for l in fr:
		jobj = json.loads(l.strip())
		if dic.get(jobj['tweet_id']) is not None:
			dic[jobj['tweet_id']]['text'] = jobj['text']
			dic[jobj['tweet_id']]['relevance_label'] = jobj['label']

# Process number of tweets posted each day.
relevant_dates_dic = {}
irrelevant_dates_dic = {}

for key, value in dic.items():
	tokens = value['created_at'].split()
	dates_key = tokens[1] + tokens[2]
	dic[key]['day_label'] = dates_key

	if value['relevance_label'] == 'relevant':
		if relevant_dates_dic.get(dates_key) is None:
			relevant_dates_dic[dates_key] = 0
		relevant_dates_dic[dates_key] += 1
	else:
		if irrelevant_dates_dic.get(dates_key) is None:
			irrelevant_dates_dic[dates_key] = 0
		irrelevant_dates_dic[dates_key] += 1

for i in sorted(relevant_dates_dic.keys()):
	print((i, relevant_dates_dic[i]), end=' ')

print('')
for i in sorted(irrelevant_dates_dic.keys()):
	print((i, irrelevant_dates_dic[i]), end=' ')

# Move relevance column to the end.
df = pd.DataFrame.from_dict(dic, orient='index')
df = df.join(df.pop('relevance_label'))

# Label encode day label.
le = preprocessing.LabelEncoder()
le.fit(sorted(df['day_label'].unique()))
df['day_label'] = le.transform(df['day_label'])

# Clean text.
text_field = 'text'
df[text_field] = df[text_field].str.replace(r'http\S+', '')
df[text_field] = df[text_field].str.replace(r'http', '')
df[text_field] = df[text_field].str.replace(r'@\S+', '')
df[text_field] = df[text_field].str.replace(r'[^A-Za-z0-9(),!?@\'\`\"\_\n]', ' ')
df[text_field] = df[text_field].str.replace(r'\'', '')
df[text_field] = df[text_field].str.replace(r'@', 'at')
df[text_field] = df[text_field].str.lower()

# Convert relevance label to integers.
df.loc[df['relevance_label'] == 'relevant', 'relevance_label'] = 0
df.loc[df['relevance_label'] == 'not_relevant', 'relevance_label'] = 1

print(df)

# Save data.
df.to_csv(output_file, index_label='id')
