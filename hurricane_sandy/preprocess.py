import pandas as pd
import json
import csv

def extract_id(text):
    return text.split(',')[1].split(':')[1]

df = pd.read_csv('release.txt', sep='\t')
df['date'] = df['date'].apply(lambda x: x.split('T')[0])

dates = ['2012-10-22', '2012-10-23', '2012-10-24', '2012-10-25', '2012-10-26', '2012-10-27',
         '2012-10-28', '2012-10-29', '2012-10-30', '2012-10-31', '2012-11-01', '2012-11-02']
dates_dfs = [pd.DataFrame(columns=[column for column in df.columns]) for date in dates]

for i in range(0, len(dates_dfs)):
    dates_dfs[i] = df[df['date'] == dates[i]]
    # dates_dfs[i] = dates_dfs[i][:40000]
    # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == True].shape[0])
    # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == False].shape[0])
    # print(' ')
    # num_true = dates_dfs[i][dates_dfs[i]['sandy_keyword'] == True]
    # sandy = num_true.sample(min(2000, num_true.shape[0]), random_state=47)
    # non_sandy = dates_dfs[i][dates_dfs[i]['sandy_keyword'] == False].sample(2000 - sandy.shape[0], random_state=47)

    # dates_dfs[i] = pd.concat([sandy, non_sandy])
    # print(dates_dfs[i].shape[0])
    # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == True].shape[0])
    # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == False].shape[0])
    dates_dfs[i] = dates_dfs[i].sample(frac=1, random_state=47)
    # print(dates_dfs[i].shape[0])
    # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == True].shape[0])
    # print(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == False].shape[0])
    # print(' ')
    # dates_dfs[i] = dates_dfs[i].sample(2000, random_state=47)

result = pd.concat(dates_dfs)
result = result.apply(lambda x: json.dumps({'tweet_id': extract_id(x['id']), 'date': x['date'], 'sandy_keyword': x['sandy_keyword']}), axis=1)
print(df.head())
result.to_csv('sample3/sampled_ids_3.json', header=False, index=False, quoting=csv.QUOTE_NONE, sep='|')

# df = pd.read_csv('release.txt', sep='\t')
# df = df.apply(lambda x: json.dumps({'tweet_id': extract_id(x['id']), 'date': x['date'], 'sandy_keyword': x['sandy_keyword']}), axis=1)
# print(df.head())
# df.to_csv('ids.json', header=False, index=False, quoting=csv.QUOTE_NONE, sep='|')