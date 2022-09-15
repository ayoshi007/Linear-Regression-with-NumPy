import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

class LinRegModel:
    def __init__(self, learn_rate, n_iter=1000, tolerance=0.001):
        self.learn_rate = learn_rate
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.weights: np.ndarray = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        n = np.size(X_train, axis=1) + 1
        data_pts = self.append_ones(X_train)
        weights = np.random.randn(1, n)
        self.weights = mse_gradient(weights, data_pts, y_train)

    def predict(self, X_test):
        self.check_fit()
        data_pts = self.append_ones(X_test)
        preds = []
        for i in range(np.size(data_pts, axis=0)):
            preds.append(np.dot(self.weights, data_pts[i]))
        return preds

    def append_ones(self, data):
        return np.append(np.ones(shape=(np.size(data, axis=0), 1)), data)

    def mse(self, y_test, predictions):
        self.check_fit()
        return mean_squared_error(y_test, predictions)

    def r_squared(self, y_test, predictions):
        self.check_fit()
        return r2_score(y_test, predictions)

    def coefs(self):
        self.check_fit()
        return self.weights[1:]

    def intercept(self):
        self.check_fit()
        return self.weights[0]

    def check_fit(self):
        if not self.check_fit():
            raise Exception("Model has not been fit")


def gradient_descent(gradient, features, targets, start_weights, learn_rate=1, n_iter=1000, tolerance=0.001):
    weights = start_weights
    for i in range(n_iter):
        print('-------- Iter', i, '--------')
        negative_grad = -learn_rate * gradient(features, targets, weights)
        print(negative_grad)
        if np.all(np.abs(negative_grad) <= tolerance):
            break
        weights += negative_grad
        print('weights:', weights)
    return weights


'''
mse = (1/2m)sum(y - ^y)^2 - sum is across all data points

mse_gradient for feature i is: (1/m)sum(y - ^y) * xi
    xi = feature i for data point
'''
'''
need to check/test
'''
def mse_gradient(data_pts, targets, weights):
    # for n features/weights and m data_pts:
    # data_pts = m * n array
    # targets = 1 * m array
    # weights = 1 * n array
    # need to return a 1 * n gradient/error values for each weight
    n = np.size(weights) # number of weights/features
    m = np.size(data_pts, axis=0) # number of data points
    datapt_gradients = np.ndarray(shape=(m, n)) # vector that will contain the gradients of the weights
    for i in range(m):
        prediction_i = np.dot(weights, data_pts[i])
        residual_i = prediction_i - targets[i]
        datapt_gradients[i] = residual_i * data_pts[i]
    return datapt_gradients.mean(axis=0)