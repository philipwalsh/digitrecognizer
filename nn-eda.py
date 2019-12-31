import pandas as pd
import numpy as np


X = pd.read_csv("excluded/combined_train.csv")

print(X['label'].value_counts())