from sklearn import ensemble
import lightgbm as lgb
MODELS = {
    'randomForest' : ensemble.RandomForestClassifier(criterion = 'gini', n_estimators = 200,
                                    max_depth=5, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, 
                                    random_state = 42, verbose = 2, oob_score=True,
                                    n_jobs = -1),
    'extraTrees' : ensemble.ExtraTreesClassifier(
        n_jobs=-1, verbose=2,n_estimators = 200),
    'gradientBoosting': ensemble.GradientBoostingClassifier(
        verbose=2,n_estimators = 200),
    'lightGBM': 'lightgbm'
}