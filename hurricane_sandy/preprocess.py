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
    dates_dfs[i] = dates_dfs[i][:40000]
    dates_dfs[i] = dates_dfs[i].sample(2000, random_state=47)
    # print(date_df[date_df['sandy_keyword'] == True].shape[0])

result = pd.concat(dates_dfs)
# print(result.head())
result = result.apply(lambda x: json.dumps({'tweet_id': extract_id(x['id']), 'date': x['date'], 'sandy_keyword': x['sandy_keyword']}), axis=1)
print(df.head())
result.to_csv('sampled_ids.json', header=False, index=False, quoting=csv.QUOTE_NONE, sep='|')

# df = pd.read_csv('release.txt', sep='\t')
# df = df.apply(lambda x: json.dumps({'tweet_id': extract_id(x['id']), 'date': x['date'], 'sandy_keyword': x['sandy_keyword']}), axis=1)
# print(df.head())
# df.to_csv('ids.json', header=False, index=False, quoting=csv.QUOTE_NONE, sep='|')