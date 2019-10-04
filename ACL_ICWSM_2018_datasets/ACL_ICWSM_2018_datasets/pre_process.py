import pandas as pd
import numpy as np

data = pd.read_csv('./nepal/2015_Nepal_Earthquake_test.csv')

data_write = data['tweet_id']
data_write.write_csv('test.csv', header=None)
