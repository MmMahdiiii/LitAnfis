import pandas as pd
from sklearn import datasets
from ucimlrepo import fetch_ucirepo
from . import Test


class Digits_UCI(Test):
    def __init__(self, *args, **kwargs) -> None:
        data = fetch_ucirepo(id=81)

        self.df = data.data.features
        self.target = data.data.targets

        super().__init__(*args, **kwargs)


class Digits(Test):
    def __init__(self, *args, **kwargs) -> None:
        train_path = "./data/unfis_data/optdigits.tra"
        self.df = pd.read_csv(train_path, header=None)

        test_path = "./data/unfis_data/optdigits.tes"
        test_data = pd.read_csv(test_path, header=None)

        self.target = 64
        super().__init__(*args, **kwargs, test_data=test_data, index=False, train_size=3823)


class Segmentation(Test):
    def __init__(self, *args, **kwargs) -> None:
        df1 = "./data/unfis_data/segmentation.data"
        df2 = "./data/unfis_data/segmentation.test"
        
        df1 = pd.read_csv(df1, header=None)
        print(df1.shape)
        df2 = pd.read_csv(df2, header=None)
        print(df2.shape)

        self.df = pd.concat([df1, df2], axis=0)           
        print(self.df.shape)

        self.df = self.df.reset_index(drop=True)
        self.df = self.df.drop(1, axis=1)

        self.target = 0
        super().__init__(*args, **kwargs)


class Diabetes(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/diabetes/diabetes.csv"
        self.df = pd.read_csv(path)
        super().__init__(*args, **kwargs)


class BCW(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/unfis_data/breast-cancer-wisconsin.data"
        self.df = pd.read_csv(path, header=None)
        self.df.drop(0, axis=1, inplace=True)
        self.target=9
        print(self.df.shape)
        super().__init__(*args, **kwargs)

class DNA(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/unfis_data/dna.arff"
        self.df = pd.read_csv(path, header=None)
        self.target=180
        print(self.df.shape)
        super().__init__(*args, **kwargs)

class Smoke(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "data/smoke/smoke_detection_iot.csv"
        self.df = pd.read_csv(path)
        self.df.drop(['index','UTC'], axis=1, inplace=True)
        self.target="Fire Alarm"
        print(self.df.shape)
        super().__init__(*args, **kwargs)

class MNIST(Test):
    def __init__(self, *args, **kwargs) -> None:
        train_path = "data/mnist/mnist_train.csv"
        self.df = pd.read_csv(train_path)

        test_path = "data/mnist/mnist_test.csv"
        test_data = pd.read_csv(test_path)

        self.target = 'label'
        
        super().__init__(*args, **kwargs, test_data=test_data, index=False)
