import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                            AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from collections import OrderedDict
from datetime import datetime

# train dataset
dataset = '/home/markg/kaggle/titanic/data/working/train_20_12_2018_1655.csv'
df = pd.read_csv(dataset, encoding='latin-1')

# split into 90% training, 10% testing
X = df.drop(columns = ['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

# # Extra Trees Parameters
# et_params = {
# 'n_jobs': [-1],
# 'n_estimators':[1000],
# 'max_depth': [3, 5, 7, 10, 12],
# 'min_samples_leaf': [1, 2, 3],
# 'min_samples_split': [2, 3, 5, 7],
# 'max_features': ['sqrt', 'log2'],
# 'verbose': [0]
# }
#
# # grid search for extra trees classifier
# et = ExtraTreesClassifier()
# grid_search = GridSearchCV(estimator=et, param_grid=et_params, cv=3, n_jobs=-1,
#                             verbose=0)
# grid_search.fit(X_train, y_train)
# print (grid_search.best_params_)

## output from grid search
# {'max_depth': 7, 'max_features': 'sqrt', 'min_samples_leaf': 3,
# 'min_samples_split': 3, 'n_estimators': 1000, 'n_jobs': -1, 'verbose': 0}


# # AdaBoost parameters
# ada_params = {
# 'n_estimators': [500, 1000],
# 'learning_rate' : [0.1, 0.2, 0.3, 0.25]
# }
#
# # grid search for Ada boost
# ad = AdaBoostClassifier()
# grid_search = GridSearchCV(estimator=ad, param_grid=ada_params, cv=3, n_jobs=-1,
#                             verbose=0)
# grid_search.fit(X_train, y_train)
# print (grid_search.best_params_)
#
# ## output
# # {'learning_rate': 0.3, 'n_estimators': 1000}

# # Gradient Boosting parameters
# gb_params = {
# 'n_estimators': [500, 1000],
# 'max_depth': [3, 5, 7, 10],
# 'min_samples_leaf': [1, 2, 3],
# 'min_samples_split': [2, 3, 4, 5],
# 'verbose': [0]
# }
#
# # grid search for Ada boost
# gb = GradientBoostingClassifier()
# grid_search = GridSearchCV(estimator=gb, param_grid=gb_params, cv=3, n_jobs=-1,
#                             verbose=0)
# grid_search.fit(X_train, y_train)
# print (grid_search.best_params_)

# ## output
# {'max_depth': 7, 'min_samples_leaf': 3, 'min_samples_split': 3,
# 'n_estimators': 500, 'verbose': 0}


# # Support Vector Classifier parameters
# svc_params = {
#     'kernel' : ['rbf', 'linear', 'sigmoid'],
#     'C' : [0.7, 0.75, 0.8, 0.85]
#     }
#
# svc = SVC()
# grid_search = GridSearchCV(estimator=svc, param_grid=svc_params, cv=3, n_jobs=-1,
#                             verbose=0)
# grid_search.fit(X_train, y_train)
# print (grid_search.best_params_)
#
# # output
# {'C': 0.7, 'kernel': 'linear'}
