import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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

    # print (data.isnull().sum())
    ########### END Missing Data #############

    # change males = 1, females = 0
    label = preprocessing.LabelEncoder()
    data['Sex'] = label.fit_transform(data['Sex'])

    # create title variable & dummy variable
    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    data.drop(columns=['Name'], inplace=True)
    title_names = (data['Title'].value_counts() < 10) #remove uncommon titles
    data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    data = pd.get_dummies(data, columns=['Title'], prefix=['title'],
    drop_first=True)

    # create a family variable
    data['FamilySize'] = data ['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 1 #initialize to yes/1 is alone
    data['IsAlone'].loc[data['FamilySize'] > 1] = 0

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

    # standardize numeric variables Age and Fare
    numeric = ['Age', 'Fare']
    data[numeric] = preprocessing.StandardScaler().fit_transform(data[numeric])

    return data

def main():

    # pre-processing training data
    dataTrain = '/home/markg/kaggle/titanic/dataset/titanicTrain.csv'
    data = pd.read_csv(dataTrain, encoding='latin-1')
    dfTrain = preprocess(data)
    dfTrain.drop(columns = ['PassengerId', 'Cabin', 'Ticket'], inplace = True)
    dfTrain.to_csv('titanicCleanTrain.csv')

    # pre-processing test data
    dataTest = '/home/markg/kaggle/titanic/dataset/titanicTest.csv'
    data = pd.read_csv(dataTest, encoding='latin-1')
    dfTest = preprocess(data)
    dfTest.drop(columns = ['Cabin', 'Ticket'], inplace = True)
    dfTest.to_csv('titanicCleanTrain.csv')

    # print data info
    print ("Train dataset: ")
    print (dfTrain.info())
    print("-"*20)
    print ("Test dataset: ")
    print (dfTest.info())

if __name__=='__main__':
    main()
