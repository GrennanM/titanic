import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():

    dataset = '/home/markg/kaggle/titanic/titanicClean.csv'
    df = pd.read_csv(dataset, encoding='latin-1')
    df = df.drop(columns = ['Unnamed: 0']) # identifier column

    # print first few rows in data and data types
    # print(df.head())
    # print (df.describe())
    # print (df.info())

    # split into 90% training, 10% testing
    X = df.drop(columns = ['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

    # build a Random Forest Model
    rf = RandomForestClassifier(n_estimators=500, max_depth=None,
     min_samples_split=2, random_state=0)

    # # fit the training model
    # rf.fit(X_train, y_train)
    #
    # # predict validation data set
    # y_pred = rf.predict(X_test)
    # print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    #
    # # print metrics and confusion matrix
    # print(classification_report(y_test,y_pred))
    # print(confusion_matrix(y_test, y_pred))

    # cross-validation
    # scores = cross_val_score(rf, X, y, cv=10, n_jobs=5)
    # print (scores.mean())

if __name__=='__main__':
    main()
