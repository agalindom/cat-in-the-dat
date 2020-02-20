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

tr = pd.read_csv('../input/train.csv')
tt = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/new_train.csv')
test = pd.read_csv('../input/new_test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
# kf = pd.read_csv('../input/KF_train.csv')

#basic eda



print(train.columns)
print(test.columns)
for i in list(train.columns):
    if i not in list(test.columns):
        print(i)

print('\n')
print('test-train')
for i in list(test.columns):
    if i not in list(train.columns):
        print(i)
# for i in list(test.columns):
#     if i not in list(kf.columns):
#         print(i)

