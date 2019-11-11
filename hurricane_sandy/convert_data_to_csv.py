import ujson as json
import pandas as pd

df = pd.DataFrame(columns=['tweet_id', 'text', 'date', 'short_date', 'geo_coordinates', 'sandy_keyword'])
dictionary = {}

with open('sample3/sampled_ids_3.json.label.json') as labels:
    for l in labels:
        jobj = json.loads(l.strip())
        dictionary[jobj['tweet_id']] = {'tweet_id': jobj['tweet_id'], 'short_date': jobj['date'], 'sandy_keyword': jobj['sandy_keyword']}

with open('sample3/sampled_ids_3.json.data.json') as data:
    for d in data:
        jobj = json.loads(d.strip())
        dic = dictionary[str(jobj['id'])] 
        dic['text'] = jobj['text']
        dic['date'] = jobj['created_at']
        if jobj['geo'] is not None:
            dic['geo_coordinates'] = list(jobj['geo']['coordinates'])
            df = df.append(dic, ignore_index=True)
        # else:
        #     dic['geo_coordinates'] = None

# for key, value in dictionary.items():
#     print(key, end=': ')
#     print(value)

# print(count)

# for key, value in dictionary.items():
#     if value['geo_coordinates'] is not None:
#         # print(value)
#         df = df.append(value, ignore_index=True)

print(df.head())
df.to_csv('processed_tweets_sample3.csv', index=False)
