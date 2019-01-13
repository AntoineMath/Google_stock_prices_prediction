import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model



def test(dataset_train_path, dataset_test_path):
    timesteps = 60
    regressor = load_model("model.h5")
    dataset_train = pd.read_csv(dataset_train_path)
    dataset_test = pd.read_csv(dataset_test_path)
    real_stock_price = dataset_test.iloc[:, 1:2].values

    # Getting the predicted stock price of 2017
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values
    inputs = inputs.reshape(-1,1)
    sc = MinMaxScaler(feature_range = (0, 1))
    inputs = sc.fit_transform(inputs)
    X_test = []
    for i in range(timesteps, timesteps + dataset_test.shape[0]):
        X_test.append(inputs[i-timesteps:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Visualising the results
    plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test("Google_Stock_Price_Train.csv","Google_Stock_Price_Test.csv")