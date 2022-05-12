import pandas as pd
import numpy as np

data1 = pd.read_csv('../data/kaggle_twitter_data.csv')
data2 = pd.read_csv('../data/dataworld_twitter_data.csv')

data1 = data1[['sentiment','tweet']]
data2 = data2[['sentiment','tweet']]

frames = [data1, data2]

data = pd.concat(frames)

positive = data[data['sentiment']==1]
print(len(positive))

negative = data[data['sentiment']==-1]
print(len(negative))

neutral = data[data['sentiment']==0]
print(len(neutral))

