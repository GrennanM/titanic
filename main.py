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
    data['Sex'] = data['Sex'].map( {'male':1, 'female':0} )

    # create n-1 dummy variables for 'Embarked' variable
    data = pd.get_dummies(data, columns=['Embarked'], prefix=['embark'],
     drop_first=True).head()

     # To do: standardize numeric variables Age and Fare


    # Alternative method of encoding using sklearn
    # encode categorical variable 'Embarked'
    # data['Embarked'] = le.fit_transform(data['Embarked'])

    # to view encoded categories
    # print (list(le.classes_))

    return data


def main():

    dataset = '/home/markg/kaggle/titanic/dataset/titanicTrain.csv'
    data = pd.read_csv(dataset, encoding='latin-1')
    df = preprocess(data)

    # print first few rows in data and data types
    # print(df.head())
    print (df.info())

if __name__=='__main__':
    main()
