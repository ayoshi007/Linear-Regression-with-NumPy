import numpy as np
from gradient_descent import gradient_descent, mse_gradient, sgd

class SelfCodedLinRegModel:
    def __init__(self, params: dict):
        self.n_iter = 1000 if 'n_iter' not in params else params['n_iter']
        self.tolerance = 0.001 if 'tolerance' not in params else params['tolerance']
        self.learn_rate = 0.0001 if 'learn_rate' not in params else params['learn_rate']
        self.weights = None

    def fit(self, X_train, y_train):
        X_train, y_train = np.array(X_train), np.array(y_train)
        weights = np.random.randn(np.size(X_train, axis=1) + 1)
        data_pts = self.__append_ones(X_train)
        self.weights = gradient_descent(mse_gradient, data_pts, y_train, weights,
                                        self.learn_rate, self.n_iter, self.tolerance)

    def predict(self, X_test):
        self.__check_fit()
        X_test = np.array(X_test)
        data_pts = self.__append_ones(X_test)
        return np.array([np.dot(i, self.weights) for i in data_pts])

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