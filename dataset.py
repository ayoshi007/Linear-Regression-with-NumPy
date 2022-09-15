import pandas as pd
from os.path import exists
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datasetdownload import download_csv

class PortugueseStudentMathGradesDataSet:
    target_var = 'G3'
    binary_cats = ['Pstatus', 'address', 'sex', 'famsize', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                   'higher', 'internet', 'romantic']
    nominal_vars = ['guardian', 'reason', 'Mjob', 'Fjob']

    def __init__(self, file, url):
        if not exists(file):
            download_csv(file, url)
        self.df = pd.read_csv(file, sep=';')
        self.preprocessed = False

    def preprocess(self):
        # dummy variables for binary attributes
        self.df = pd.get_dummies(data=self.df, columns=self.binary_cats, drop_first=True)
        # dummy variables for non-binary categorical attributes
        self.df = pd.get_dummies(data=self.df, columns=self.nominal_vars)
        # Series of absolute correlations, descending order
        corrs = self.df.corr()[self.target_var].abs().sort_values(ascending=False)
        # get top 15 correlations, plus the target variable
        self.df = self.df[corrs.index[:16]]

        # get DataFrame with attributes
        no_target = self.df.drop(self.target_var, axis=1)

        scaler = MinMaxScaler()

        # scale the attributes
        X = scaler.fit_transform(no_target)
        # create DataFrame out of scaled attributes
        scaled_df = pd.DataFrame(data=X, columns=no_target.columns)
        # append target variable to scaled DataFrame
        scaled_df[self.target_var] = self.df[self.target_var]
        # store scaled DataFrame to instance variable
        self.df = scaled_df
        self.preprocessed = True

    def get_split(self, test_size=0.2, random=None):
        if not self.preprocessed:
            raise Exception("Data has not been preprocessed yet")
        X = self.df.drop(columns=self.target_var, axis=1)
        y = self.df[self.target_var]
        return train_test_split(X, y, test_size=test_size, random_state=random)

    def get_target_range(self):
        if not self.preprocessed:
            raise Exception("Data has not been preprocessed yet")
        return self.df['G3'].min(), self.df['G3'].max()
