import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def preprocess(data):
    # count the number of missing values
    # print (data.isnull().sum())

    # impute mean in place of missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    # drop 2 missing values from Embarked
    data.dropna(subset=['Embarked'], inplace=True)

    # drop columns: Name, ID, Cabin, Ticket
    data.drop(columns = ['Name', 'PassengerId', 'Cabin', 'Ticket'],
     inplace = True)

    # change males = 1, females = 0
    label = preprocessing.LabelEncoder()
    data['Sex'] = label.fit_transform(data['Sex'])

    # create n-1 dummy variables for 'Embarked' variable
    data = pd.get_dummies(data, columns=['Embarked'], prefix=['embark'],
     drop_first=True)

    # topcode Fare variable at 100
    for i in range(0, len(data['Fare'])):
        try:
            if int(data['Fare'][i]) >= 100:
                data.at[i, 'Fare'] = 100
        except KeyError:
            pass
            # print ("KeyError Caught")

     # histogram of Fare
    # data['Fare'].plot.hist(grid=True, rwidth=0.9, color='#607c8e')
    # plt.show()

    # standardize numeric variables Age and Fare
    numeric = ['Age', 'Fare']
    data[numeric] = preprocessing.StandardScaler().fit_transform(data[numeric])

    return data

def main():

    dataset = '/home/markg/kaggle/titanic/dataset/titanicTrain.csv'
    data = pd.read_csv(dataset, encoding='latin-1')
    df = preprocess(data)
    df.to_csv('titanicClean.csv')

    # print first few rows in data and data types
    # print(df.head())
    # # print (df.describe())
    # # print (df.info())

if __name__=='__main__':
    main()
