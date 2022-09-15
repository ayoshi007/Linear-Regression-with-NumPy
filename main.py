from dataset import PortugueseStudentMathGradesDataSet
from linreg_model import LinRegModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

dataset = PortugueseStudentMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

dataset.preprocess()
X_train, X_test, y_train, y_test = dataset.get_split()

model = LinRegModel(learn_rate=0.0001, n_iter=1000, tolerance=1e-06)
model.fit(X_train, y_train)