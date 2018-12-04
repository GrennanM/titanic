import pandas as pd
import numpy as np
from sklearn import preprocessing

def main():
    ############### PRE-PROCESSING ################################
    dataset = '/home/markg/kaggle/titanic/dataset/titanicTrain.csv'
    data = pd.read_csv(dataset, encoding='latin-1')

    # label encoder
    le = preprocessing.LabelEncoder()

    #### MISSING DATA ######
    # count the number of missing values
    # print (data.isnull().sum())

    # impute mean in place of missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    # drop 2 missing values from Embarked
    data.dropna(subset=['Embarked'], inplace=True)

    # drop columns: Name, ID,
    data.drop(columns = ['Name', 'PassengerId', 'Cabin'], inplace = True)

    # change males = 1, females = 0
    data['Sex'] = data['Sex'].map( {'male':1, 'female':0} )

    # encode categorical variable 'Embarked'
    # data['Embarked'] = le.fit_transform(data['Embarked'])

    # create n-1 dummy variables for 'Embarked' variable
    data = pd.get_dummies(data, columns=['Embarked'], prefix=['embark'],
     drop_first=True).head()

    # to view encoded categories
    # print (list(le.classes_))

    # print first few rows in data
    print(data.head())

    # prints data types
    print (data.info())

    # describes data
    # print(data.describe())

if __name__=='__main__':
    main()
