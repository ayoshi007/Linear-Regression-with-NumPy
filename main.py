from dataset import PortugueseStudentMathGradesDataSet
from linreg_model import gradient_descent, mse_gradient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# dataset = PortugueseStudentMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')
#
# dataset.preprocess()
# X_train, X_test, y_train, y_test = dataset.get_split()
#
# print(X_train.to_numpy().shape)
# print(y_train.to_numpy().shape)

weights = np.array([0.5,0.5])
# x = np.array([[1,5], [1,15], [1,25], [1,35], [1,45], [1,55]])
# y = np.array([5, 20, 14, 32, 22, 38])
x = np.array([[1,1], [1,10], [1,120], [1,12], [1,6.45]])
y = np.array([11, 65, 725, 77, 43.7])
print(weights)
weights = gradient_descent(mse_gradient, x, y, weights, learn_rate=0.0000001, n_iter=100000)

print('final weights:', weights)
