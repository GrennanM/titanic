import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # train dataset
    dataset = '/home/markg/kaggle/titanic/titanicCleanTraining.csv'
    df = pd.read_csv(dataset, encoding='latin-1')
    df = df.drop(columns = ['Unnamed: 0']) # identifier column

    # # test dataset
    # dt = '/home/markg/kaggle/titanic/titanicCleanTest.csv'
    # test_df = pd.read_csv(dt, encoding='latin-1')
    # test_df = test_df.drop(columns = ['Unna   med: 0']) # identifier column
    #
    # # prepare passengerID for submission file
    # passengerID = test_df['PassengerId']
    # test_df.drop(columns = ['PassengerId'], inplace = True)
    #
    # print first few rows in data and data types
    # print(df.head())
    # print (df.describe())
    # print (df.info())

    # split into 80% training, 20% testing
    X = df.drop(columns = ['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # build a Random Forest Model
    rf = RandomForestClassifier(n_estimators=500, max_depth=None,
     min_samples_split=2, random_state=0)

    # # fit the training model
    rf.fit(X_train, y_train)

    # predict validation data set
    y_pred = rf.predict(X_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    #
    # # print metrics and confusion matrix
    # print(classification_report(y_test,y_pred))
    # print(confusion_matrix(y_test, y_pred))

    # cross-validation
    # scores = cross_val_score(rf, X, y, cv=10, n_jobs=5)
    # print (scores.mean())

    # fit Random Forest for submission
    # rf.fit(X, y)
    # predictions = rf.predict(test_df)
    # create submission file
    # submission = pd.DataFrame({'PassengerId':passengerID, 'Survived':predictions})
    # filename = 'titanicPredictions1.csv'
    # submission.to_csv(filename, index=False)

if __name__=='__main__':
    main()
