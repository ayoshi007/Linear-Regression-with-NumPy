import numpy as np

def gradient_descent(gradient, features, targets, start_weights, learn_rate=1, n_iter=1000, tolerance=0.001):
    weights = start_weights
    for i in range(n_iter):
        negative_grad = -learn_rate * np.array(gradient(features, targets, weights))
        if np.all(np.abs(negative_grad) <= tolerance):
            break
        weights += negative_grad
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
    # get the residuals for each data point prediction for each data point with current weights
    print(np.shape(data_pts))
    res = np.array([np.dot(i, weights) for i in data_pts]) - targets
    # get the gradient by taking the mean of the sum between each data point column and their residual
    return [(column * res).mean() for column in data_pts.transpose()]