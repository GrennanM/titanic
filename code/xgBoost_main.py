import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def main():
    # train dataset
    dataset = '/home/markg/kaggle/titanic/data/working/train_20_12_2018_1239.csv'
    df = pd.read_csv(dataset, encoding='latin-1')

    # split into 90% training, 10% testing
    X = df.drop(columns = ['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

    # # fit baseline model to training data
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # # make predictions for test data
    # y_pred = xgb_model.predict(X_test)
    #
    # # evaluate predictions
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # cross-validation
    scores = cross_val_score(xgb_model, X_train, y_train, cv=10, n_jobs=-1)
    print ("Accuracy: ", scores.mean())

    # print model parameters
    # print (xgb_model)

    # ######### Hyper-Parameter Tuning using Random and Grid Search #############
    #
    # # Parameters to search
    # random_grid = {'max_depth': [2, 4, 5, 6, 10],
    #                 'min_child_weight': [1, 2, 3],
    #                 'gamma': [0, 0.1, 0.2],
    #                 'colsample_bytree': [0.8, 1]}
    #
    # # create model
    # xgb = XGBClassifier()
    #
    # # random search
    # # xgb_random = RandomizedSearchCV(estimator = xgb,
    # # param_distributions = random_grid,
    # #  n_iter = 100, cv = 3, verbose=0, random_state=42, n_jobs = -1)
    #
    # # grid search
    # xgb_random = GridSearchCV(estimator = xgb,
    # param_grid = random_grid, cv = 3, verbose=0, n_jobs = -1)
    #
    # # Fit the random search model
    # xgb_random.fit(X_train, y_train)
    #
    # print ("Best parameters:", xgb_random.best_params_)
    #
    # ####################### END Tuning #####################################

    # # test dataset
    # dt = '/home/markg/kaggle/titanic/titanicCleanTest.csv'
    # test_df = pd.read_csv(dt, encoding='latin-1')
    # test_df = test_df.drop(columns = ['Unnamed: 0']) # identifier column
    #
    # # prepare passengerID for submission file
    # passengerID = test_df['PassengerId']
    # test_df.drop(columns = ['PassengerId'], inplace = True)
    #
    # # fit Random Forest for submission
    # xgb_model.fit(X, y)
    # predictions = xgb_model.predict(test_df)
    #
    # # create submission file
    # submission = pd.DataFrame({'PassengerId':passengerID, 'Survived':predictions})
    # filename = 'titanicXGBpredictions1.csv'
    # submission.to_csv(filename, index=False)

if __name__=='__main__':
    main()
