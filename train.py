import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from os import makedirs
from os.path import exists, abspath

#Change this variable : this the number of timesteps used by the model to make the prediction
Graph = abspath("Graph")
if not exists(Graph):
    makedirs(Graph)

def train(csv_train):
    timesteps = 60
    # Importing the training set
    dataset_train = pd.read_csv(csv_train)
    training_set = dataset_train.iloc[:, 1:2].values #taking the 2nd column (1:2 because upperbound excluded)

    # Feature Scaling (Normalisation)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []


    #sliding widows of 60 timesteps with a stride of 1, over the dataset
    for i in range(timesteps,dataset_train.shape[0]): 
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping (number_of_stockprices, timesteps, number of indicators)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



    # Part 2 - Building the RNN
    regressor = Sequential()

    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))

    # output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    tbCallback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32, callbacks = [tbCallback])
    regressor.save("model.h5")

if __name__ == "__main__":
    train("Google_Stock_Price_Train.csv")