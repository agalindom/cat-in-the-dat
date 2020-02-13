import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
from sklearn import model_selection
from sklearn import ensemble 
import lightgbm as lgb 
import feature_generator as fg
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#load test set
test = pd.read_csv('../input/new_test.csv')
test = test.drop('id', axis = 1)
sub = pd.read_csv('../input/sample_submission.csv')
# load model
clf = joblib.load('../models/random_forest.pkl')

# Make predictions
preds = clf.predict_proba(test)[:,1]

# create submission file
sub['target'] = preds
sub.to_csv('../output/submission_rf.csv', index = False)

