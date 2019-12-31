import pandas as pd
import numpy as py

train = pd.read_csv("excluded/train.csv")

col_min = 0
col_max = 777
for x in range(col_min, col_max):
    train['pixel'+str(x)] = train['pixel'+str(x+5)]
for x in range(783, 778):
    train['pixel'+str(x)] = 0
train.to_csv('excluded/shifted-left.csv', index=False)

del train

train = pd.read_csv("excluded/train.csv")
col_min = 5
col_max = 783

for x in range(col_max, col_min, -1):
    train['pixel'+str(x)] = train['pixel'+str(x-5)]

for x in range(0, 4):
    train['pixel'+str(x)] = 0

train.to_csv('excluded/shifted-right.csv', index=False)


del train

train_df = pd.read_csv("excluded/train.csv")
left_df = pd.read_csv("excluded/shifted-left.csv")
right_df = pd.read_csv("excluded/shifted-right.csv")

combined_df = pd.concat([train_df, left_df, right_df], axis=0)
combined_df.to_csv('excluded/combined_train.csv',index=0)
