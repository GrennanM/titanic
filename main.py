import pandas as pd
import numpy as np

def main():
    ############### PRE-PROCESSING ################################
    dataset = '/home/markg/kaggle/titanic/dataset/titanicTrain.csv'
    data = pd.read_csv(dataset, encoding='latin-1')

    # drop columns: Name, ID,
    data.drop(columns = ['Name', 'PassengerId', 'Cabin'], inplace = True)

    # change males = 1, females = 0
    data['Sex'] = data['Sex'].map( {'male':1, 'female':0} )

    # change

    # count the number of missing values
    # print (data.isnull().sum())

    # impute mean in place of missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    # drop 2 missing values from Embarked
    data.dropna(subset=['Embarked'], inplace=True)

    # print first few rows in data
    print(data.head())

    # prints data types
    data.info()

    # describes data
    print(data.describe())

if __name__=='__main__':
    main()
