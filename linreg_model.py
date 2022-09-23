import numpy as np

'''
Please note that this model crashes for learn_rates greater than 0.4
'''

def gradient_descent(gradient, features, targets, start_weights, learn_rate=1, n_iter=1000, tolerance=0.001):
    weights = start_weights
    for i in range(n_iter):
        # get gradient 'step'
        negative_grad = -learn_rate * np.array(gradient(features, targets, weights))
        # terminate if below tolerance
        if np.all(np.abs(negative_grad) <= tolerance):
            break
        # make gradient 'step'
        weights += negative_grad
    # return fitted weights
    return weights


def mse_gradient(data_pts, targets, weights) -> list:
    # get the residuals for each data point prediction for each data point with current weights
    res = np.array([np.dot(i, weights) for i in data_pts]) - targets
    # get the gradient by taking the mean of the sum between each data point column and their residual
    return [(column * res).mean() for column in data_pts.transpose()]

class SelfCodedLinRegModel:
    def __init__(self, params):
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