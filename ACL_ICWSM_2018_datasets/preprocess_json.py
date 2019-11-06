import pandas as pd
import numpy as np
import csv
import json

dic = {}
with open('./data/ids.json.data.json') as fr:
	for l in fr:
		jobj = json.loads(l.strip())
		dic[jobj['id_str']] = {'created_at': jobj['created_at']}

with open('./data/ids.json.label.json') as fr:
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
# dates = ['2012-10-22', '2012-10-23', '2012-10-24', '2012-10-25', '2012-10-26', '2012-10-27',
#          '2012-10-28', '2012-10-29', '2012-10-30', '2012-10-31', '2012-11-01', '2012-11-02']
# dates_dfs = [pd.DataFrame(columns=[column for column in df.columns]) for date in dates]

# for i in range(0, len(dates_dfs)):
#     dates_dfs[i] = df[df['date'] == dates[i]]
#     dates_dfs[i] = dates_dfs[i][:40000]
#     # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == True].shape[0])
#     # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == False].shape[0])
#     # print(' ')
#     # num_true = dates_dfs[i][dates_dfs[i]['sandy_keyword'] == True]
#     # sandy = num_true.sample(min(2000, num_true.shape[0]), random_state=47)
#     # non_sandy = dates_dfs[i][dates_dfs[i]['sandy_keyword'] == False].sample(2000 - sandy.shape[0], random_state=47)

#     # dates_dfs[i] = pd.concat([sandy, non_sandy])
#     # print(dates_dfs[i].shape[0])
#     # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == True].shape[0])
#     # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == False].shape[0])
#     dates_dfs[i] = dates_dfs[i].sample(5000, random_state=47)
#     # print(dates_dfs[i].shape[0])
#     # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == True].shape[0])
#     # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == False].shape[0])
#     # print(' ')
#     # dates_dfs[i] = dates_dfs[i].sample(2000, random_state=47)
