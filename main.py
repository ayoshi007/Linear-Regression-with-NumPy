from dataset import PortugueseStudentMathGradesDataSet
from linreg_model import LinRegModel

# fetching data
dataset = PortugueseStudentMathGradesDataSet('student-mat.csv', 'https://personal.utdallas.edu/~afy180000/intro_ml/assignment1/')

# hyperparameters
learn_rate = 0.000001
n_iter = 10000
tolerance = 1e-06
test_size = 0.2

# getting data ready
dataset.preprocess()
X_train, X_test, y_train, y_test = dataset.get_split(test_size=test_size)
min_y, max_y = dataset.get_target_range()

# training model and getting predictions
model = LinRegModel(learn_rate=learn_rate, n_iter=n_iter, tolerance=tolerance)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# evaluation metrics
mse = model.mse(y_test, preds)
r2_score = model.r_squared(y_test, preds)
intercept = model.intercept()

with open('parameter_log.csv', 'a') as file:
    file.write(f'{learn_rate},{n_iter},{tolerance},{test_size},{min_y},{max_y},{mse},{r2_score},{intercept}\n')

