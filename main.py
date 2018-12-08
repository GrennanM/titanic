import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def main():

    dataset = '/home/markg/kaggle/titanic/titanicClean.csv'
    df = pd.read_csv(dataset, encoding='latin-1')
    df = df.drop(columns = ['Unnamed: 0']) # identifier column

    # print first few rows in data and data types
    print(df.head())
    # print (df.describe())
    # print (df.info())

    # split into 90% training, 10% testing
    X = df.drop(columns = ['Survived'])
    Y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10)

if __name__=='__main__':
    main()
