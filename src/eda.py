import pandas as pd 
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
import seaborn as sns 
sns.set(style="darkgrid")
import warnings
import statsmodels.api as sm
warnings.filterwarnings("ignore")

train = pd.read_csv('../input/new_train.csv')
test = pd.read_csv('../input/new_test.csv')
sub = pd.read_csv('../input/sample_submission.csv')

#basic eda

print(train.shape)
print(train.head())
print('\n')
print(test.shape)
print(test.head())
print(train.isnull().sum())
print('\n')
print(test.isnull().sum())

