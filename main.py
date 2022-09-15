from dataset import PortugueseStudentMathGradesDataSet
from linreg_model import LinRegModel
import time as tm
from itertools import product

# fetching data
dataset = PortugueseStudentMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

# hyperparameter to test
learn_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
iters = [100, 1000, 10000, 100000, 1000000]
tolerances = [1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09]
test_sizes = [0.1, 0.2]

# getting data ready
dataset.preprocess()
min_y, max_y = dataset.get_target_range()

# go through each combination of hyperparameters
for test_size, tolerance, n_iter, learn_rate in product(test_sizes, tolerances, iters, learn_rates):
    # get training data and test data
    X_train, X_test, y_train, y_test = dataset.get_split(test_size=test_size)

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

    with open('parameter_log_self_coded.csv', 'a') as file:
        file.write(
            f'{learn_rate},{n_iter},{tolerance},{test_size},{min_y},{max_y},{mse},{r2_score},{intercept},{training_time}s\n')


