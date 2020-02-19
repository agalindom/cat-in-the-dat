import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
from sklearn import model_selection
from sklearn import ensemble 
import lightgbm as lgb 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import lightgbm as lgb

from . import dispatcher

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
FOLD = int(os.environ.get('FOLD')) # Fold must be an int
MODEL = os.environ.get('MODEL')

FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

if __name__ == '__main__':
    # Load datasets
    df = pd.read_csv(TRAINING_DATA)

    print(len(df.columns))
    print(len(df.columns))
    ########## Partition ############### 
    # create train and validation sets from dictionary
    train = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop = True)
    valid = df[df.kfold==FOLD].reset_index(drop = True)

    #drop id variable from train
    train = train.drop('id', axis = 1)
    valid = valid.drop('id', axis = 1)

    # separate target and training data
    ytrain = train['target']
    yvalid = valid['target']
    train_df = train.drop(['target', 'kfold'], axis = 1)
    valid_df = valid.drop(['target', 'kfold'], axis = 1)

    # Make sure valid and train have the same order of columns
    valid_df = valid_df[train_df.columns]

    train_data = lgb.Dataset(train_df, ytrain)
    valid_data = lgb.Dataset(valid_df, ytrain, reference=train_data)

    params = {
                    'learning_rate': 0.05,
                    'feature_fraction': 0.1,
                    'min_data_in_leaf' : 12,
                    'max_depth': 3,
                    'reg_alpha': 1,
                    'reg_lambda': 1,
                    'objective': 'binary',
                    'metric': 'auc',
                    'n_jobs': -1,
                    'n_estimators' : 4000,
                    'feature_fraction_seed': 42,
                    'bagging_seed': 42,
                    'boosting_type': 'gbdt',
                    'verbose': 1,
                    'is_unbalance': True,
                    'boost_from_average': False}

    # Initialize model
    if MODEL == 
    clf.train(
        params=params,
        train_set=train_data,
        early_stopping_rounds=20,
        valid_sets=[valid_data],
        eval_metric='auc'
        verbose_eval=100)
    preds = clf.predict_proba(valid_df)[:,1]
    print(metrics.roc_auc_score(yvalid, preds))

    ## Save Model for prediction
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")

