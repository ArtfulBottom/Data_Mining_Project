import pandas as pd
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim

# tweets = []
# with open('../data/training.json.data.json') as fr:
# 	for l in fr:
# 		jobj = json.loads(l.strip())
# 		if (jobj['geo'] != None):
# 			# print(jobj['geo'])
# 			tweets.append(jobj)

mp = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, projection='lcc', lat_1=32, lat_2=45, lon_0=-95)
mp.readshapefile('st99_d00', name='states', drawbounds=True)

geolocator = Nominatim()
# loc = geolocator.geocode('New York')
# x, y = mp(70, 40)
# print(x)
# print(y)

# mp.plot(x, y, marker='o', color='red', markersize=10)
dic = {}
with open('../data/training.json.data.json') as fr:
	for l in fr:
		jobj = json.loads(l.strip())
		if jobj['geo'] is not None:
			dic[jobj['id_str']] = [jobj['geo']['coordinates']]

with open('../data/training.json.label.json') as fr:
	for l in fr:
		jobj = json.loads(l.strip())
		if dic.get(jobj['tweet_id']) is not None:
			dic[jobj['tweet_id']].append(jobj['relevance'])

for key, value in dic.items():
	x, y = mp(value[0][1], value[0][0])
	if value[1]:
		mp.plot(x, y, marker='o', color='blue', markersize=2)
	else:
		mp.plot(x, y, marker='o', color='red', markersize=2)

plt.show()
