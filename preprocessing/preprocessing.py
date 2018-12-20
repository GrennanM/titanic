import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn import preprocessing
from datetime import datetime

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
    #         # print ("KeyError Caught")

    # Apply log to Fare to reduce skewness distribution
    data["Fare"] = data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

    # standardize numeric variables Age and Fare
    numeric = ['Age', 'Fare']
    preprocessing.StandardScaler().fit_transform(data[numeric])

    return data

def main():

    # read original data and pre-processing training data
    dataTrain = '/home/markg/kaggle/titanic/data/original/titanicTrain.csv'
    data = pd.read_csv(dataTrain, encoding='latin-1') # read original data
    dfTrain = preprocess(data)
    dfTrain.drop(columns = ['PassengerId', 'Cabin', 'Ticket'], inplace = True)

    # write working training data to csv
    training_path = '/home/markg/kaggle/titanic/data/working/'
    data_train_filename = 'train_' + str(datetime.now().strftime('%d_%m_%Y_%H%M')) + '.csv'
    dfTrain.to_csv(training_path + data_train_filename, index=False)

    # # read original data and pre-processing test data
    dataTest = '/home/markg/kaggle/titanic/data/original/titanicTest.csv'
    data = pd.read_csv(dataTest, encoding='latin-1')
    dfTest = preprocess(data)
    dfTest.drop(columns = ['Cabin', 'Ticket'], inplace = True)

    # write working test data to csv
    test_path = '/home/markg/kaggle/titanic/data/working/'
    data_test_filename = 'test_' + str(datetime.now().strftime('%d_%m_%Y_%H%M')) + '.csv'
    dfTest.to_csv(test_path + data_test_filename, index=False)

    # print data info
    print ("Train dataset: ")
    print (dfTrain.info())
    print("-"*30)
    print ("Test dataset: ")
    print (dfTest.info())

if __name__=='__main__':
    main()
