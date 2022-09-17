from portugalmathgradesdataset import PortugalMathGradesDataSet
from linreg_model import SelfCodedLinRegModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product

learn_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
n_iters = [1000, 10000, 100000, 1000000]
tolerances = [1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10]

# fetching data
print('Fetching data...')
dataset = PortugalMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

# getting data ready
dataset.preprocess()
min_y, max_y = dataset.get_target_range()
test_size = 0.2
X_train, X_test, y_train, y_test = dataset.get_split(test_size)

repeats = 1
total = len(learn_rates) * len(n_iters) * len(tolerances) * repeats
counter = 1

file = open('parameter_log_self_coded.csv', 'a')

print('Iterating through', total, 'hyperparameter varations...')
for n_iter, learn_rate, tolerance in product(n_iters, learn_rates, tolerances):
    hyper_params = {
        'learn_rate': learn_rate,
        'n_iter': n_iter,
        'tolerance': tolerance
    }
    for _ in range(repeats):
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

        # print()
        # print("=============== RESULTS ===============")
        # print('--------Hyperparameters--------')
        # print('Learning rate:', hyper_params['learn_rate'])
        # print('Num. iters:', hyper_params['n_iter'])
        # print('Tolerance:', hyper_params['tolerance'])
        # print('----------Error Stats----------')
        # print(f'\t\tTrain\t\t\t\t\tTest')
        # print(f"RMSE:\t{mse_train ** 0.5}\t\t{mse_test ** 0.5}")
        # print(f"MSE:\t{mse_train}\t\t{mse_test}")
        # print(f"MAE:\t{mae_train}\t\t{mae_test}")
        # print(f"R^2:\t{r2_train}\t\t{r2_test}")
        # print('Weights')
        # print(model.weights)

        file.write(f'{hyper_params["learn_rate"]},{hyper_params["n_iter"]},{hyper_params["tolerance"]},'
                   f'{min_y},{max_y},{mse_train ** 0.5},{mse_test ** 0.5},{mse_train},{mse_test},'
                   f'{mae_train},{mae_test},{r2_train},{r2_test},')
        for weight in model.weights:
            file.write(f'{weight}')
            if weight != model.weights[-1]:
                file.write(',')
        file.write('\n')
        print(f'{counter}/{total}: Recorded in log')
        counter += 1

print('Done')
file.close()