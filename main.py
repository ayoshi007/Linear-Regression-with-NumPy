from dataset import PortugueseStudentMathGradesDataSet
from linreg_model import LinRegModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

dataset = PortugueseStudentMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

dataset.preprocess()
X_train, X_test, y_train, y_test = dataset.get_split()

learn_rate = 0.0000001
n_iter = 10000
tolerance = 1e-06
model = LinRegModel(learn_rate=learn_rate, n_iter=n_iter, tolerance=tolerance)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = model.mse(y_test, preds)
r2_score = model.r_squared(y_test, preds)
intercept = model.intercept()
coefs = model.coefs()

with open('parameter_log.csv', 'a') as file:
    file.write(f'{learn_rate},{n_iter},{tolerance},{mse},{r2_score},{intercept},{coefs}')