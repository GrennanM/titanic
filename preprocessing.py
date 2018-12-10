import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def preprocess(data):
    ########## Missing Data ###############
    # count the number of missing values
    # print (data['Fare'].isnull().sum())

    # impute mean in place of missing values for Age
    data['Age'].fillna(data['Age'].median(), inplace=True)

    # impute median in place of missing value in Fare
    data['Fare'].fillna(data['Fare'].median(), inplace = True)

    # impute mode for Embarked
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

    # drop columns: Name, ID, Cabin, Ticket
    data.drop(columns = ['PassengerId', 'Cabin', 'Ticket', 'Name'], inplace = True)

    # print (data.isnull().sum())
    ########### END Missing Data #############

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
    print (df.head())
    df.to_csv('titanicCleanTrain.csv')

    # print first few rows in data and data types
    # print(df.head())
    # # print (df.describe())
    # # print (df.info())

if __name__=='__main__':
    main()
