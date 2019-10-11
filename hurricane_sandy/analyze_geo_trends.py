import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap

def identify_min_max_coordinates(coordinate_list):
    lats = []
    lons = []
    for coordinates in coordinate_list:
        coordinates = coordinates.strip('[]').split(',')
        lats.append(float(coordinates[0]))
        lons.append(float(coordinates[1]))
    return lats, lons


df = pd.read_csv('processed_tweets.csv', lineterminator='\n')
lats, lons = identify_min_max_coordinates(df['geo_coordinates'])
hmap = folium.Map()
hm_wide = HeatMap(list(zip(lats, lons, [1] * len(lats))), min_opacity=0.2, max_val=1, blur=15, radius=10)
hmap.add_child(hm_wide)
hmap
