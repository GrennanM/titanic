import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                            AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from datetime import datetime

# train dataset
dataset = '/home/markg/kaggle/titanic/data/working/train_20_12_2018_1655.csv'
train = pd.read_csv(dataset, encoding='latin-1')

# test dataset
dt = '/home/markg/kaggle/titanic/data/working/test_20_12_2018_1655.csv'
test = pd.read_csv(dt, encoding='latin-1')

# parameters for later
ntrain = train.shape[0] # number of entries/rows in train
ntest = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, random_state=SEED)
PassengerId = test['PassengerId'] # Store passenger ID for submission
test.drop(columns = ['PassengerId'], inplace=True)

# helper class
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x,y).feature_importances_

# get out of fold predictions
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,)) # 839 zeros in array
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
     # (-1, 1)  reshapes to a single column
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# parameters for models
# parameters for random forest
rf_params = {
'n_jobs': -1,
'n_estimators': 1000,
'warm_start': True,
'max_features': 'sqrt',
'max_depth': 5,
'min_samples_leaf': 3,
'min_samples_split': 10,
'verbose': 0
}

# Extra Trees Parameters
et_params = {
'n_jobs': -1,
'n_estimators':1000,
'max_features': 'sqrt',
'max_depth': 7,
'min_samples_leaf': 3,
'verbose': 0,
'min_samples_split': 3
}

# AdaBoost parameters
ada_params = {
'n_estimators': 1000,
'learning_rate' : 0.3
}

# Gradient Boosting parameters
gb_params = {
'n_estimators': 500,
'max_depth': 7,
'min_samples_leaf': 3,
'min_samples_split': 3,
'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.75
    }

# generate objects to represent models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and response to feed into models
y_train = train['Survived'].ravel() # flattens response to a 1D array
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an np array of the training data
x_test = test.values # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # SVC

# # feature importance
# rf_feature = rf.feature_importances(x_train,y_train)
# et_feature = et.feature_importances(x_train, y_train)
# ada_feature = ada.feature_importances(x_train, y_train)
# gb_feature = gb.feature_importances(x_train,y_train)
#
# # rffeatures=list(rf_feature)

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel(),
      'SVC': svc_oof_train.ravel(),
    })
# print (base_predictions_train.head())

# create dataset for second level
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train,
                        gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test,
                            gb_oof_test, svc_oof_test), axis=1)

# second level model
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

# create submission file
submission = pd.DataFrame({'PassengerId':PassengerId, 'Survived':predictions})
path = '/home/markg/kaggle/titanic/data/submissions/'
filename = 'stacking_submission_' + str(datetime.now().strftime('%d_%m_%Y_%H%M')) + '.csv'
submission.to_csv(path+filename, index=False)
