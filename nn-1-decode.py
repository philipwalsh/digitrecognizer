import pandas as pd
import numpy as py

preds = pd.read_csv("excluded/nn-predictions.csv")

preds.columns = ['ImageId', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

preds['ImageId'] += 1
preds['Label'] = 0


print(preds.head(15))

preds.loc[preds['1']>0.5, 'Label']=1
preds.loc[preds['2']>0.5, 'Label']=2
preds.loc[preds['3']>0.5, 'Label']=3
preds.loc[preds['4']>0.5, 'Label']=4
preds.loc[preds['5']>0.5, 'Label']=5
preds.loc[preds['6']>0.5, 'Label']=6
preds.loc[preds['7']>0.5, 'Label']=7
preds.loc[preds['8']>0.5, 'Label']=8
preds.loc[preds['9']>0.5, 'Label']=9

#preds[ (preds['2']>.5)]['Label'] = 2


drop_cols= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for col in drop_cols:
    preds.drop(col, axis=1, inplace=True)

print(preds.head(15))
preds.to_csv('excluded/nn-submission.csv', index=False)
print('\n***')
print('*** complete')
print('***')