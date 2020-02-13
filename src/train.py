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

# Load datasets
train = pd.read_csv('../input/new_train.csv')
test = pd.read_csv('../input/new_test.csv')

print(len(train.columns))
print(len(test.columns))
########## Partition ############### 
## shuffle the data
train = train.sample(frac=1).reset_index(drop=True)

#drop id variable from train
train = train.drop('id', axis = 1)

#save test id
test_id = test['id']
test = test.drop('id', axis = 1)

# separate target and training data
y = train['target']
X = train.drop('target', axis = 1)

## basic train test_split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.2)


########## Model Creation ############### 
# RandomForestClassifier(criterion='gini', 
#                              n_estimators=700,
#                              min_samples_split=10,
#                              min_samples_leaf=1,
#                              max_features='auto',
#                              oob_score=True,
#                              random_state=1,
#                              n_jobs=-1)


clf = ensemble.RandomForestClassifier(criterion = 'gini', n_estimators = 1000,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   random_state = 42, verbose = 2, oob_score=True,
                                   n_jobs = -1)

clf.fit(X_train, y_train)

## predict on validation set
preds = clf.predict_proba(X_test)[:,1]

## Save Model for prediction
joblib.dump(clf, "../models/random_forest.pkl")

## Feature importances
print(metrics.roc_auc_score(y_test, preds))
features = train.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

fig, axs = plt.subplots(figsize = (15,16))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

