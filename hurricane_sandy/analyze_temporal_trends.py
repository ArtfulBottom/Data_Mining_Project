import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def count_keywords(keywords, texts):
    count = 0
    for text in texts:
        text = text.lower()
        for keyword in keywords:
            if keyword in text:
                count += 1
    return count

df = pd.read_csv('processed_tweets.csv', lineterminator='\n')
print(df.shape[0])

dates = ['2012-10-22', '2012-10-23', '2012-10-24', '2012-10-25', '2012-10-26', '2012-10-27',
         '2012-10-28', '2012-10-29', '2012-10-30', '2012-10-31', '2012-11-01', '2012-11-02']
date_abbrvs = ['10-22', '10-23', '10-24', '10-25', '10-26', '10-27', 
               '10-28', '10-29', '10-30', '10-31', '11-01', '11-02']
dates_dfs = [pd.DataFrame(columns=[column for column in df.columns]) for date in dates]
num_sandy = []
num_hurricane = []
keywords = ['sandy', 'hurricane', 'storm', 'injured', '#sandy', '#hurricane', '#hurricanesandy', 'windy', 'flood']
num_keywords = []

for i in range(0, len(dates_dfs)):
    dates_dfs[i] = df[df['short_date'] == dates[i]]
    num_sandy.append(dates_dfs[i][dates_dfs[i]['sandy_keyword'] == True].shape[0])
    num_hurricane.append(count_keywords(['hurricane'], dates_dfs[i]['text']))
    num_keywords.append(count_keywords(keywords, dates_dfs[i]['text']))

for i, date in enumerate(dates):
    print('%s\nNum sandy: %d, Num hurricane: %d, Num keywords: %d' % (date, num_sandy[i],
            num_hurricane[i], num_keywords[i]), end='\n\n')

index = np.arange(len(dates))
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index, num_hurricane, bar_width, alpha=opacity, edgecolor='white', color='blue', label='sandy')
rects2 = plt.bar(index + bar_width, num_sandy, bar_width, alpha=opacity, edgecolor='white', color='purple', label='hurricane')
rects3 = plt.bar(index + 2 * bar_width, num_keywords, bar_width, alpha=opacity, edgecolor='white', color='orange', label='Hurricane disaster keywords')

plt.xlabel('Date (year 2012)')
plt.ylabel('Number of tweets with keyword(s)')
plt.title('Temporal Trends for Relevant (Disaster-related) Keywords')
plt.xticks(index + bar_width , date_abbrvs)
plt.legend()

plt.show()

# for date_df in dates_dfs:
#     print(date_df.head())
