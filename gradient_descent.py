import numpy as np
import random

def gradient_descent(gradient, features, targets, start_weights, learn_rate=1, n_iter=1000, tolerance=0.001, batch_size=1):
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


def sgd(gradient, features, targets, start_weights, learn_rate=1, n_iter=1000, tolerance=0.001, batch_size=1):
    data = np.insert(features, features.shape[1], targets, axis=1)
    np.random.shuffle(data)
    features = data[:, :-1]
    targets = data[:, -1]
    weights = start_weights
    for _ in range(n_iter):
        i = random.randint(0, targets.size - batch_size)
        batch_features = features[i:batch_size + i]
        batch_targets = targets[i:batch_size + i]
        negative_grad = -learn_rate * np.array(gradient(batch_features, batch_targets, weights))
        if np.all(np.abs(negative_grad) <= tolerance):
            break
        weights += negative_grad
    # return fitted weights
    return weights
