import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from collections import OrderedDict

def main():
    # train dataset
    dataset = '/home/markg/kaggle/titanic/titanicCleanTrain.csv'
    df = pd.read_csv(dataset, encoding='latin-1')
    df = df.drop(columns = ['Unnamed: 0']) # identifier column

    # print first few rows in data and data types
    # print(df.head())
    # print (df.describe())

    # split into 80% training, 20% testing
    X = df.drop(columns = ['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

    # # build a Random Forest Model
    # random_grid_model = RandomForestClassifier(n_estimators=1400, max_depth=10,
    #  min_samples_split=10, random_state=0, min_samples_leaf=4,
    #  max_features='log2', bootstrap=False)

    # Accuracy = 0.8309
    grid_search_model = RandomForestClassifier(n_estimators=1500, max_depth=5,
    max_features='sqrt', min_samples_leaf=3, min_samples_split=10,
    bootstrap=True)

    # grid_search_model.fit(X_train, y_train)

    # # fit the training model
    # baseline_model.fit(X_train, y_train)
    #
    # # predict validation data set
    # y_pred = baseline_model.predict(X_test)
    # print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    # cross-validation
    scores = cross_val_score(grid_search_model, X, y, cv=10, n_jobs=-1)
    print ("Accuracy: ", scores.mean())

    # ################# Hyper-Parameter Tuning #########################
    # ###### Randomized Search #############
    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 10)]
    # # Number of features to consider at every split
    # max_features = ['sqrt', 'log2', None]
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    #
    # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    #
    # # First create the base model to tune
    # rf = RandomForestClassifier()
    # # Random search of parameters, using 3 fold cross validation,
    # # search across 100 different combinations, and use all available cores
    # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
    #  n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    #
    # # Fit the random search model
    # rf_random.fit(X_train, y_train)
    #
    # print ("Best parameters:", rf_random.best_params_)
    #
    # ######### END Randomized Search ##################

    # ##############  Grid Search ##########
    # # # Number of trees in random forest
    # # n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2500, num = 10)]
    # #
    # # # create grid for tuning
    # # grid_from_random_search = {'n_estimators': 2000,
    # # 'min_samples_split': 10, 'min_samples_leaf': 2,
    # # 'max_features': None, 'max_depth': 40, 'bootstrap': True}
    #
    # param_grid = {'n_estimators': [1000],
    # 'min_samples_split': [8, 9, 10, 11, 12],
    # 'min_samples_leaf': [3, 4, 5],
    # 'max_features': [3, 4, 5, 6],
    # 'max_depth': [5, 10, 15],
    # 'bootstrap': [True, False]}
    #
    # # model
    # rf = RandomForestClassifier()
    #
    # # Grid Search of parameters, using 3 fold cross validation
    # grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
    #                       cv = 3, n_jobs = -1, verbose = 2)
    #
    # # Fit the random search model
    # grid_search.fit(X_train, y_train)
    # print (grid_search.best_params_)
    # ############## End Grid Search ###############

    # ################ print OOB vs search criteria #############################
    # RANDOM_STATE = 123
    #
    # # NOTE: Setting the `warm_start` construction parameter to `True` disables
    # # support for parallelized ensembles but is necessary for tracking the OOB
    # # error trajectory during training.
    # ensemble_clfs = [
    #     ("RandomForestClassifier, max_features='sqrt'",
    #         RandomForestClassifier(n_estimators=100,
    #                                warm_start=True, oob_score=True,
    #                                max_features="sqrt",
    #                                random_state=RANDOM_STATE)),
    #     ("RandomForestClassifier, max_features='log2'",
    #         RandomForestClassifier(n_estimators=100,
    #                                warm_start=True, max_features='log2',
    #                                oob_score=True,
    #                                random_state=RANDOM_STATE)),
    #     ("RandomForestClassifier, max_features=None",
    #         RandomForestClassifier(n_estimators=100,
    #                                warm_start=True, max_features=None,
    #                                oob_score=True,
    #                                random_state=RANDOM_STATE))
    # ]
    #
    # # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    # error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    #
    # # Range of `n_estimators` values to explore.
    # min_estimators = 15
    # max_estimators = 200 # number of trees to try
    #
    # for label, clf in ensemble_clfs:
    #     for i in range(min_estimators, max_estimators + 1, 10):
    #         clf.set_params(n_estimators=i)
    #         clf.fit(X_train, y_train)
    #
    #         # Record the OOB error for each `n_estimators=i` setting.
    #         oob_error = 1 - clf.oob_score_
    #         error_rate[label].append((i, oob_error))
    #
    # # Generate the "OOB error rate" vs. "n_estimators" plot.
    # for label, clf_err in error_rate.items():
    #     xs, ys = zip(*clf_err)
    #     plt.plot(xs, ys, label=label)
    #
    # axes = plt.gca()
    # axes.set_ylim([0,0.4]) # sets y_axis limit to between 0 and 1
    # plt.xlim(min_estimators, max_estimators)
    # plt.xlabel("n_estimators")
    # plt.ylabel("OOB error rate")
    # plt.legend(loc="upper right")
    # plt.show()
    # ######################## END PLOT ###########################


    # #################### Feature Importance Plot ##################
    # imyportances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(grid_search_model.feature_importances_,3)})
    # importances = importances.sort_values('importance',
    # ascending=False).set_index('feature')
    #
    # print (importances.head(20))


    # Build a forest and compute the feature importances
    # forest = ExtraTreesClassifier(n_estimators=1000,
    #                               random_state=0)
    #
    # forest.fit(X_train, y_train)
    # importances = forest.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_],
    #              axis=0)
    # indices = np.argsort(importances)[::-1]
    #
    # # Print the feature ranking
    # print("Feature ranking:")
    # for f in range(X.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #
    # print (X.info())

    # # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(X.shape[1]), importances[indices],
    #        color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X.shape[1]), indices)
    # plt.xlim([-1, X.shape[1]])
    # plt.show()
    # ###################### End Feature Importance Plot ######################

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
    # grid_search_model.fit(X, y)
    # predictions = grid_search_model.predict(test_df)
    #
    # # create submission file
    # submission = pd.DataFrame({'PassengerId':passengerID, 'Survived':predictions})
    # filename = 'titanicPredictions4.csv'
    # submission.to_csv(filename, index=False)

if __name__=='__main__':
    main()
