from portugalmathgradesdataset import PortugalMathGradesDataSet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
from itertools import product


'''
part2.py

This file uses the scikit-learn SGDRegressor to perform linear regression on the dataset.
The dataset can be found here: https://archive.ics.uci.edu/ml/datasets/Student+Performance

The file will download the dataset and iterate through the all combinations of the specified hyperparameters.
Each time a model is fit, a different train-test split is made.
The hyperparameters, error metrics, and weights of each run is recorded in the output file.
The error metrics taken are:
    Root mean squared error
    Mean squared error
    Mean absolute error
    R^2 score

The user can add/remove values to the learning_rates, eta0s, max_iters, tols, and early_stoppings lists
    to add more hyperparameters to test.
The repeats integer variable is the amount of times each hyperparameter combination is ran.
'''

# output file
filename = 'library_output1.csv'
file_columns = 'learn_rate_type,learn_rate,n_iter,tolerance,min_y,max_y,rmse_train,rmse_test,mse_train,mse_test,mae_train,mae_test,R^2_train,R^2_test,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,early_stopping\n'

# best params from testing are:
# learn_rate = 'adaptive', eta0 = 0.9, max_iter = 10000, tol = 1e-06, early_stopping = False

# learning_rates must either be: 'adaptive', 'constant', 'optimal', 'invscaling'
learning_rates = ['adaptive']
eta0s = [0.9]
max_iters = [10000]
tols = [1e-06]
early_stoppings = [False]
repeats = 1

# fetching dataset
print('Fetching data...')
dataset = PortugalMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

# getting data ready
dataset.preprocess()
min_y, max_y = dataset.get_target_range()
test_size = 0.2

# setting number of runs
total = len(learning_rates) * len(eta0s) * len(max_iters) * len(tols) * len(early_stoppings) * repeats
counter = 1

# setting up output file
file = open(filename, 'a')
if not file.tell():
    file.write(file_columns)

# runs
print('Iterating through', total, 'runs...')
for max_iter, learning_rate, eta0, tol, early_stopping in product(max_iters, learning_rates, eta0s, tols, early_stoppings):
    hyper_params = {
        'learning_rate': learning_rate,
        'eta0': eta0,
        'max_iter': max_iter,
        'tol': tol,
        'early_stopping': early_stopping
    }
    for _ in range(repeats):
        X_train, X_test, y_train, y_test = dataset.get_split(test_size)
        model = SGDRegressor(learning_rate=hyper_params['learning_rate'],
                             eta0=hyper_params['eta0'],
                             max_iter=hyper_params['max_iter'],
                             tol=hyper_params['tol'])
        model.fit(X_train, y_train)
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)

        mse_train = mean_squared_error(y_train, preds_train)
        mse_test = mean_squared_error(y_test, preds_test)
        mae_train = mean_absolute_error(y_train, preds_train)
        mae_test = mean_absolute_error(y_test, preds_test)
        r2_train = r2_score(y_train, preds_train)
        r2_test = r2_score(y_test, preds_test)
        weights = list(model.coef_)
        weights.insert(0, model.intercept_[0])

        file.write(
            '{learning_rate},{eta0},{max_iter},{tol},{min_y},{max_y},{rmse_train},{rmse_test},{mse_train},{mse_test},{mae_train},{mae_test},{r2_train},{r2_test},'.format(
            learning_rate=hyper_params["learning_rate"], eta0=hyper_params["eta0"], max_iter=hyper_params['max_iter'], tol=hyper_params['tol'],
            min_y=min_y, max_y=max_y, rmse_train=mse_train ** 0.5, rmse_test=mse_test ** 0.5, mse_train=mse_train,
            mse_test=mse_test,
            mae_train=mae_train, mae_test=mae_test, r2_train=r2_train, r2_test=r2_test
        ))

        for weight in weights:
            file.write('{weight},'.format(weight=weight))
        file.write('{early_stopping}'.format(early_stopping=hyper_params["early_stopping"]))
        file.write('\n')
        print('Run {counter}/{total}: Recorded in log'.format(counter=counter, total=total))
        counter += 1


print('Done')
file.close()
