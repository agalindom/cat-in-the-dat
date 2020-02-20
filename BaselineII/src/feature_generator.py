import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
import seaborn as sns 
sns.set(style="darkgrid")
import warnings
import statsmodels.api as sm
from tqdm import tqdm_notebook as tqdm
from sklearn import model_selection
import category_encoders as ce
warnings.filterwarnings("ignore")

## load the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

####################################################################
############### Encoding functions ############################
####################################################################

def label_encode(data, feats):
    df = data.copy(deep = True)
    dict_of_dicts = {}
    for col in feats:
        length = []
        values = []
        encoder_dict = {}
        for idx, val in zip(range(len(df[col].unique())),df[col].unique()):
            if str(val) == 'nan':
                pass
            else:
                length.append(idx)
                values.append(val)
        for l,v in zip(length, values):
            encoder_dict[v] = l

        dict_of_dicts[col] = encoder_dict

        df[col] = df[col].replace(dict_of_dicts.get(col))
        
    return df

def CountEncoding(df, cols, df_test=None):
    for col in cols:
        frequencies = df[col].value_counts().reset_index()
        df_values = df[[col]].merge(frequencies, how='left', left_on=col, right_on='index').iloc[:,-1].values
        df[col+'_counts'] = df_values
        df_test_values = df_test[[col]].merge(frequencies, how='left', left_on=col, right_on='index').fillna(1).iloc[:,-1].values
        df_test[col+'_counts'] = df_test_values
    count_cols = [col+'_counts' for col in cols]
    return df, df_test

def TargetEncoder(train, test, smoothing = 0.3):
    # get binary columns
    train.sort_index(inplace=True)
    target = train['target']
    test_id = test['id']
    train.drop(['target', 'id'], axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)
    cat_feats = train.columns.tolist()
    smoothing = 0.3

    oof = pd.DataFrame([])

    for train_idx, valid_idx in model_selection.StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(train, target):
        tgt_encoder = ce.TargetEncoder(cols=cat_feats, smoothing=smoothing)
        tgt_encoder.fit(train[cat_feats].iloc[train_idx], target[train_idx])
        oof = oof.append(tgt_encoder.transform(train.iloc[valid_idx, :]), ignore_index=False)

    tgt_encoder = ce.TargetEncoder(cols = cat_feats, smoothing=smoothing)
    tgt_encoder.fit(train, target)
    train = oof.sort_index()
    test = tgt_encoder.transform(test)
    train['target'] = target
    test['id'] = test_id
    print('Target encoding done!')
    return train, test


####################################################################
############### Binary preprocessing function ######################
####################################################################

def binary_processor(data):
    df = data.copy(deep = True)

    # get binary columns
    binary = []
    for col in df.columns:
        if 'bin' in col:
            binary.append(col)
            
    #label encoder for binary feats
    print('progress bar for label encoder: \n')
    df = label_encode(df, feats = ['bin_3', 'bin_4'])
    
    # # bin_0
    # df['bin_0'] = df['bin_0'].replace(np.nan, df['bin_0'].value_counts().index[0])
    # # bin_1
    # df['bin_1'] = df['bin_1'].replace(np.nan, df['bin_1'].value_counts().index[0])
    # # bin_2
    # df['bin_2'] = df['bin_2'].replace(np.nan, df['bin_2'].value_counts().index[0])
    # # bin_3
    # df['bin_3'] = df['bin_3'].replace(np.nan, df['bin_3'].value_counts().index[0])
    # # bin_4
    # df['bin_4'] = df['bin_4'].replace(np.nan, df['bin_4'].value_counts().index[0])
    
    
    # sanity check
    #sanity_check
    print('missing values for binary cols')
    for col in binary:
        print(f'{col} null values: {df[col].isnull().sum()}/{df.shape[0]} - \
    prop: {round(df[col].isnull().sum()/df.shape[0], 3)}')
    
    return df

####################################################################
############### mean encoding functions ############################
####################################################################

def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()
    
    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()
    
    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)
    
    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values

def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = model_selection.KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index)
    
    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
      
        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)
        
        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature       
    return train_feature.values

def mean_target_encoding(train, test, target, categorical, alpha=5):
  
    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)
  
    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)
    
    # Return new features to add to the model
    return train_feature, test_feature



####################################################################
############### Nominal preprocessing function #####################
####################################################################

def nominal_processor1(data, nominal):
    df = data.copy(deep = True)

    # label encode nominal columns
    print('progress bar for label encoder: \n')
    df = label_encode(df, feats = nominal)
    
    # imputation
    # for col in nominal:
    #     df[col] = df[col].replace(np.nan, df[col].value_counts().index[0])
    
    print('missing values for nominal cols')
    for col in nominal:
        print(f'{col} null values: {df[col].isnull().sum()}/{df.shape[0]} - \
    prop: {round(df[col].isnull().sum()/df.shape[0], 3)}')
        
    # for nom columns 0 through 4 one hot encode will do
    # one hot encoding
    #df = pd.get_dummies(df, columns = nominal[:5])
    
    return df

def full_nominal_processor(train_data, test_data):
    df_train = train_data.copy(deep = True)
    df_test = test_data.copy(deep = True)

    df_train['set'] = 'train'
    df_test['set'] = 'test'

    all_data = pd.concat([df_train, df_test])

    nominal = []
    for col in all_data.columns:
        if 'nom' in col:
            nominal.append(col)
    
    df_train, df_test = CountEncoding(df_train, nominal, df_test = df_test)
    
    print('encoder: ')
    all_data = nominal_processor1(all_data, nominal)

    df_train = all_data.query("set == 'train'").drop('set', axis = 1)
    df_test = all_data.query("set == 'test'").drop('set', axis = 1)
    df_test.drop('target', axis =1, inplace=True)

    # for nominals 6 through 7 a mean encoding will have to be as for the high cardinality variables


    return df_train, df_test


####################################################################
############### Ordinal preprocessing function #####################
####################################################################

def ordinal_processor(train, test):
    df_train = train.copy(deep = True)
    df_test = test.copy(deep = True)

    ordinal = []
    for col in df_train.columns:
        if 'ord' in col:
            ordinal.append(col)

    df_train, df_test = CountEncoding(df_train, ordinal, df_test = df_test)

    ordinal = []
    for col in df_train.columns:
        if 'ord' in col:
            ordinal.append(col)
    ####### create replacing dictionaries for each ordinal column #########
    # 1-3 ordinal cols
    ord_1_dict = {'Novice':0, 'Contributor':1, 'Expert':2, 'Master':3, 'Grandmaster':4}
    ord_2_dict = {'Freezing':0, 'Cold':1, 'Warm':2, 'Hot':3, 'Boiling Hot':4, 'Lava Hot':5}
    ord_3_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10,
                  'l':11, 'm':12, 'n':13, 'o':14}
    
    # 4th ordinal column dictionary
    list4 = list((set(df_train['ord_4'].unique())))
    ord_4_dict = {}
    for idx,v in enumerate(list4):
        ord_4_dict[v] = idx
    
    # fifth ordinal column dictionary
    list5 = list((set(df_train['ord_5'].unique())))
    ord_5_dict = {}
    for idx,v in enumerate(list5):
        ord_5_dict[v] = idx
    
    ## replace values with dictionaries
    ordinal_dictionaries = [ord_1_dict,ord_2_dict,ord_3_dict,ord_4_dict,ord_5_dict] 
    for col,ord_dict in zip(ordinal[1:], ordinal_dictionaries):
        df_train[col] = df_train[col].replace(ord_dict)
        df_test[col] = df_test[col].replace(ord_dict)
        
    ####### impute missing values #############
    ### train
    # for col in ordinal:
    #     df_train[col] = df_train[col].replace(np.nan, df_train[col].value_counts().index[0])
    ### sanity checker
    print('missing values for train ordinal cols')
    for col in ordinal:
        print(f'{col} null values: {df_train[col].isnull().sum()}/{df_train.shape[0]} - \
    prop: {round(df_train[col].isnull().sum()/df_train.shape[0], 3)}')
    
    ### test
    # for col in ordinal:
    #     df_test[col] = df_test[col].replace(np.nan, df_test[col].value_counts().index[0])
    ### sanity checker   
    print('missing values for test ordinal cols')
    for col in ordinal:
        print(f'{col} null values: {df_test[col].isnull().sum()}/{df_test.shape[0]} - \
    prop: {round(df_test[col].isnull().sum()/df_test.shape[0], 3)}')

    # mean encoding
    # for col in ordinal:
    #     df_train[f'{col}_enc'], df_test[f'{col}_enc'] = mean_target_encoding(train=df_train,
    #                                                                  test=df_test,
    #                                                                  target='target',
    #                                                                  categorical=col,
    #                                                                  alpha=10)

   
    return df_train, df_test

####################################################################
############### Cyclical preprocessing function #####################
####################################################################

def cyclical_processor(train, test):
    df_train = train.copy(deep = True)
    df_test = test.copy(deep = True)

    for col in ['day', 'month']:
        df_test[col] = df_test[col].replace(np.nan, df_test[col].value_counts().index[0])
    for col in ['day', 'month']:
        df_train[col] = df_train[col].replace(np.nan, df_train[col].value_counts().index[0])

    ## train data
    df_train['month_sin'] = np.sin((df_train['month'] - 1) * (2.0 * np.pi / 12))
    df_train['month_cos'] = np.cos((df_train['month'] - 1) * (2.0 * np.pi / 12))

    df_train['day_sin'] = np.sin((df_train['day'] - 1) * (2.0 * np.pi / 7))
    df_train['day_cos'] = np.cos((df_train['day'] - 1) * (2.0 * np.pi / 7))

    # for col in ['month_sin', 'month_cos', 'day_sin', 'day_cos']:
    #     df_train[col] = df_train[col].replace(np.nan, df_train[col].mean())
    
    ## test data
    df_test['month_sin'] = np.sin((df_test['month'] - 1) * (2.0 * np.pi / 12))
    df_test['month_cos'] = np.cos((df_test['month'] - 1) * (2.0 * np.pi / 12))

    df_test['day_sin'] = np.sin((df_test['day'] - 1) * (2.0 * np.pi / 7))
    df_test['day_cos'] = np.cos((df_test['day'] - 1) * (2.0 * np.pi / 7))

    # for col in ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'day', 'month']:
    #     df_test[col] = df_test[col].replace(np.nan, df_test[col].mean())

    return df_train, df_test




if __name__ == '__main__':
    ########## Preprocessing ############### 
    # Binary
    train = binary_processor(train) 
    test = binary_processor(test)
    # Nominal
    train, test = full_nominal_processor(train, test)
    # Ordinal
    train, test = ordinal_processor(train,test)
    #cyclical processor
    train, test = cyclical_processor(train,test)
    # print(len(train.columns))
    # print(train.columns)
    # print(len(test.columns))
    # print(test.columns)
    #target encoder
    train = train.fillna(-1)
    train = train.fillna(-1)
    train, test = TargetEncoder(train, test)

    ## drop day month for now
    dr = []
    for i in list(train.columns):
        if i not in list(test.columns):
            dr.append(i)
    if len(dr) > 0:
        print(dr)
    else:
        print('all cool')
        
    dr = []
    for i in list(test.columns):
        if i not in list(train.columns):
            dr.append(i)
    if len(dr) > 0:
        print(dr)
    else:
        print('all cool')

    train.to_csv('../input/new_train.csv', index = False)
    test.to_csv('../input/new_test.csv', index = False)