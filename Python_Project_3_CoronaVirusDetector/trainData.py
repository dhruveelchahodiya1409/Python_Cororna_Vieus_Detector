import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
    shuffled = np.random.permutation(len(data))
    test_data_size = int(len(data) * ratio)
    test_data = shuffled[:test_data_size]
    train_data = shuffled[test_data_size:]
    return data.iloc[train_data],data.iloc[test_data]


if __name__ == '__main__':

    # Read CSV file
    df = pd.read_csv('data.csv')

    # split train and test Data
    train, test = data_split(df, 0.2)


    # create numpy array for train data and test data except infectionProb
    X_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    X_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()


    # numpy array for infectionProb
    Y_train = train[['infectionProb']].to_numpy().reshape(2400, )
    Y_test = test[['infectionProb']].to_numpy().reshape(599, )

    # Using LogisticRegression for prediction
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    #open file where you can save data
    with open("model.pkl","wb") as file:
        #dump info to that file
        pickle.dump(clf,file)

