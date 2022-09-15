from dataset import PortugueseStudentMathGradesDataSet
from linreg_model import gradient_descent, mse_gradient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

dataset = PortugueseStudentMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

dataset.preprocess()
X_train, X_test, y_train, y_test = dataset.get_split()

print(X_train.to_numpy().shape)
print(y_train.to_numpy().shape)

