#%%
import pylab
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
testdata= pd.read_csv('C:/Users/daheekim/Desktop/datacamp_data/bikesharing/test.csv')
traindata = pd.read_csv('C:/Users/daheekim/Desktop/datacamp_data/bikesharing/train.csv')
#%%
print(traindata.shape)
print(traindata.head(3))
print(traindata.info())
#print(traindata.isnull().sum())
#print(traindata.dtypes)

# %%
