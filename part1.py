from portugalmathgradesdataset import PortugalMathGradesDataSet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from linreg_model import SelfCodedLinRegModel
from itertools import product


'''
part1.py

This file performs the SelfCodedLinRegModel implementation of linear regression on the dataset.
The dataset can be found here: https://archive.ics.uci.edu/ml/datasets/Student+Performance

The file will download the dataset and iterate through the all combinations of the specified hyperparameters.
Each time a model is fit, a different train-test split is made.
The hyperparameters, error metrics, and weights of each run is recorded in the output file.
The error metrics taken are:
    Root mean squared error
    Mean squared error
    Mean absolute error
    R^2 score

The user can add/remove values to the learn_rates, n_iters, and tolerances lists
    to add more hyperparameters to test.
The repeats integer variable is the amount of times each hyperparameter combination is ran.

Please note that this model will crash is learn_rate is set to greater than 0.4
'''

# output file
filename = 'selfcoded_output1.csv'
file_columns = 'learn_rate,n_iter,tolerance,min_y,max_y,rmse_train,rmse_test,mse_train,mse_test,mae_train,mae_test,R^2_train,R^2_test,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15\n'

# best approximate params from testing:
# learn_rate = 0.08, n_iter = 5000, tolerance = 4e-06
learn_rates = [0.08]
n_iters = [5000]
tolerances = [4e-06]
repeats = 1

# fetching dataset
print('Fetching data...')
dataset = PortugalMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

# getting data ready
dataset.preprocess()
min_y, max_y = dataset.get_target_range()
test_size = 0.2

# setting number of runs
total = len(learn_rates) * len(n_iters) * len(tolerances) * repeats
counter = 1

# setting up output file
file = open(filename, 'a')
if not file.tell():
    file.write(file_columns)

# runs
print('Iterating through', total, 'runs...')
for n_iter, learn_rate, tolerance in product(n_iters, learn_rates, tolerances):
    hyper_params = {
        'learn_rate': learn_rate,
        'n_iter': n_iter,
        'tolerance': tolerance
    }
    for _ in range(repeats):
        X_train, X_test, y_train, y_test = dataset.get_split(test_size)
        model = SelfCodedLinRegModel(hyper_params)
        model.fit(X_train, y_train)
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)

        mse_train = mean_squared_error(y_train, preds_train)
        mse_test = mean_squared_error(y_test, preds_test)
        mae_train = mean_absolute_error(y_train, preds_train)
        mae_test = mean_absolute_error(y_test, preds_test)
        r2_train = r2_score(y_train, preds_train)
        r2_test = r2_score(y_test, preds_test)

        file.write(
            '{learn_rate},{n_iter},{tolerance},{min_y},{max_y},{rmse_train},{rmse_test},{mse_train},{mse_test},{mae_train},{mae_test},{r2_train},{r2_test},'.format(
            learn_rate=hyper_params["learn_rate"], n_iter=hyper_params["n_iter"], tolerance=hyper_params["tolerance"],
            min_y=min_y, max_y=max_y, rmse_train=mse_train ** 0.5, rmse_test=mse_test ** 0.5, mse_train=mse_train, mse_test=mse_test,
            mae_train=mae_train, mae_test=mae_test, r2_train=r2_train, r2_test=r2_test
        ))
        for weight in model.weights:
            file.write('{weight}'.format(weight=weight))
            if weight != model.weights[-1]:
                file.write(',')
        file.write('\n')
        print('Run {counter}/{total}: Recorded in log'.format(counter=counter, total=total))
        counter += 1

print('Done')
file.close()
