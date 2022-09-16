from dataset import PortugueseStudentMathGradesDataSet
from linreg_model import LinRegModel
import time as tm
from itertools import product
import pandas as pd

# fetching data
dataset = PortugueseStudentMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

# hyperparameters to test
learn_rates = [0.0001, 0.001, 0.01, 0.1]
iters = [100, 1000, 10000]
tolerances = [1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09]

# getting data ready
dataset.preprocess()
min_y, max_y = dataset.get_target_range()

test_size = 0.2
# get training data and test data
X_train, X_test, y_train, y_test = dataset.get_split(test_size=test_size)

file = open('parameter_log_self_coded.csv', 'w')
file.write('learn_rate,n_iter,tolerance,test_size,min_y,max_y,mse,r2_score,intercept,training_time\n')

print('Evaluating model with various hyperparameters. Please wait...')

# go through each combination of hyperparameters
for tolerance, n_iter, learn_rate in product(tolerances, iters, learn_rates):
    # training model and getting predictions
    model = LinRegModel(learn_rate=learn_rate, n_iter=n_iter, tolerance=tolerance)
    start = tm.time()
    model.fit(X_train, y_train)
    training_time = tm.time() - start
    preds = model.predict(X_test)

    # evaluation metrics
    mse = model.mse(y_test, preds)
    r2_score = model.r_squared(y_test, preds)
    intercept = model.intercept()

    file.write(
        f'{learn_rate},{n_iter},{tolerance},{test_size},{min_y},{max_y},{mse},{r2_score},{intercept},{training_time}s\n'
    )


file.close()

print('Finished iterating through all hyperparameter combinations.'
      'Finding the best combinations for MSE and R^2 score.')
tm.sleep(2)

df = pd.read_csv('parameter_log_self_coded.csv')
print('The best hyperparameters for MSE:')
print(df.iloc[df['mse'].argmin()])
print()
print('The best hyperparameters for R^2 score:')
print(df.iloc[df['r2_score'].argmax()])
print()