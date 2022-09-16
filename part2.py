from portugalmathgradesdataset import PortugalMathGradesDataSet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
from itertools import product

learning_rates = ['constant', 'adaptive', 'invscaling', 'optimal']
eta0s = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
max_iters = [10000, 100000, 1000000, 10000000, 100000000]
tols = [1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10]
early_stoppings = [False, True]


# fetching data
print('Fetching data...')
dataset = PortugalMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

# getting data ready
dataset.preprocess()
min_y, max_y = dataset.get_target_range()
test_size = 0.2
X_train, X_test, y_train, y_test = dataset.get_split(test_size)

repeats = 3
total = len(learning_rates) * len(eta0s) * len(max_iters) * len(tols) * len(early_stoppings) * repeats
counter = 1

file = open('parameter_log_library.csv', 'a')

print('Iterating through', total, 'hyperparameter varations...')
for learning_rate, eta0, max_iter, tol, early_stopping in product(learning_rates, eta0s, max_iters, tols, early_stoppings):
    hyper_params = {
        'learning_rate': learning_rate,
        'eta0': eta0,
        'max_iter': max_iter,
        'tol': tol,
        'early_stopping': early_stopping
    }
    for _ in range(repeats):
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

        # print()
        # print("=============== RESULTS ===============")
        # print('--------Hyperparameters--------')
        # print('Learning rate type:', hyper_params['learning_rate'])
        # print('eta0:', hyper_params['eta0'])
        # print('Num. iters:', hyper_params['max_iter'])
        # print('Tolerance:', hyper_params['tol'])
        # print('Early Stopping:', hyper_params['early_stopping'])
        # print('----------Error Stats----------')
        # print(f'\t\tTrain\t\t\t\t\tTest')
        # print(f"RMSE:\t{mse_train ** 0.5}\t\t{mse_test ** 0.5}")
        # print(f"MSE:\t{mse_train}\t\t{mse_test}")
        # print(f"MAE:\t{mae_train}\t\t{mae_test}")
        # print(f"R^2:\t{r2_train}\t\t{r2_test}")
        # print('Weights')
        # print(weights)


        file.write(f'{hyper_params["learning_rate"]},{hyper_params["eta0"]},{hyper_params["max_iter"]},{hyper_params["tol"]},'
                   f'{min_y},{max_y},{mse_train ** 0.5},{mse_test ** 0.5},{mse_train},{mse_test},'
                   f'{mae_train},{mae_test},{r2_train},{r2_test},')
        for weight in weights:
            file.write(f'{weight},')
        file.write(f'{hyper_params["early_stopping"]}')
        file.write('\n')
        print(f'{counter}/{total}: Recorded in log')
        counter += 1


print('Done')
file.close()
