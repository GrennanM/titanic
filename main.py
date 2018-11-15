import pandas as pd
import numpy as np

def main():
    ############### PRE-PROCESSING ################################
    dataset = '/home/markg/kaggle/titanic/dataset/titanicTrain.csv'
    data = pd.read_csv(dataset, encoding='latin-1')

    # drop columns: Name, ID,
    data.drop(columns = ['Name', 'PassengerId'], inplace = True)

    # change males = 1, females = 0
    data['Sex'] = data['Sex'].map( {'male':1, 'female':0} )

    # data.info()
    print(data.head())

if __name__=='__main__':
    main()
