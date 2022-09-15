import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from vanillagd import gradient_descent, mse_gradient
from sklearn.metrics import mean_squared_error, r2_score

class LinRegModel:
    def __init__(self, learn_rate, n_iter=1000, tolerance=0.001):
        self.learn_rate = learn_rate
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.weights = None

    def fit(self, X_train, y_train):
        X_train, y_train = np.array(X_train), np.array(y_train)
        weights = np.random.randn(np.size(X_train, axis=1) + 1)
        data_pts = self.__append_ones(X_train)
        self.weights = gradient_descent(mse_gradient, data_pts, y_train, weights,
                                        self.learn_rate, self.n_iter, self.tolerance)

    def predict(self, X_test):
        self.__check_fit()
        data_pts = self.__append_ones(X_test)
        preds = []
        for i in range(np.size(data_pts, axis=0)):
            preds.append(np.dot(self.weights, data_pts[i]))
        return preds

    def mse(self, y_test, predictions):
        self.__check_fit()
        return mean_squared_error(y_test, predictions)

    def r_squared(self, y_test, predictions):
        self.__check_fit()
        return r2_score(y_test, predictions)

    def coefs(self):
        self.__check_fit()
        return self.weights[1:]

    def intercept(self):
        self.__check_fit()
        return self.weights[0]

    def __append_ones(self, data):
        return np.insert(data, 0, values=1, axis=1)

    def __check_fit(self):
        if self.weights is None:
            raise Exception("Model has not been fit")

