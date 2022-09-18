from portugalmathgradesdataset import PortugalMathGradesDataSet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
from itertools import product

# best params from testing are:
# learn_rate = 'adaptive', eta0 = 0.95, max_iter = 15000, tol = 1e-06, early_stopping = False
filename = 'log_library_final.csv'
learning_rates = ['adaptive']
eta0s = [0.9, 0.95, 1.0, 1.05, 1.1]
max_iters = [10000, 15000, 20000, 25000]
tols = [1e-06, 1e-07, 1e-08]
early_stoppings = [False]
repeats = 20
# fetching data
print('Fetching data...')
dataset = PortugalMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

# getting data ready
dataset.preprocess()
min_y, max_y = dataset.get_target_range()
test_size = 0.2


total = len(learning_rates) * len(eta0s) * len(max_iters) * len(tols) * len(early_stoppings) * repeats
counter = 1

file = open(filename, 'a')

print('Iterating through', total, 'hyperparameter varations...')
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
