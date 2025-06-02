import pandas as pd
from sklearn import datasets
from . import Test
from ucimlrepo import fetch_ucirepo


class Cryotheraphy(Test):
    def __init__(self, *args, **kwargs) -> None:
        file_path = './data/cryotherapy/Cryotherapy.xlsx'
        self.df = pd.read_excel(file_path)
        self.target = 'Result_of_Treatment'
        super().__init__(train_size=63, *args, **kwargs)


class Haberman(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/haberman/haberman.data"
        self.df = pd.read_csv(path, header=None)
        self.target = 3
        super().__init__(train_size=214, *args, **kwargs)


class Heart(Test):
    def __init__(self, *args, **kwargs) -> None:
        data = fetch_ucirepo(id=45)

        # Assuming self.df is your features DataFrame and self.target is a DataFrame
        self.df = data.data.features
        self.target = data.data.targets

        target_column_name = self.target.columns[0]
        data = pd.concat([self.df, self.target], axis=1)
        data = data.dropna(subset=[target_column_name])

        self.df = data.iloc[:, :-1]
        self.target = data.iloc[:, -1]  # All rows, the last column
        self.target[self.target > 0] = 1
        super().__init__(train_size=189, *args, **kwargs)


class Glass(Test):
    def __init__(self, *args, **kwargs) -> None:
        data = fetch_ucirepo(id=42) 

        # Assuming self.df is your features DataFrame and self.target is a DataFrame
        self.df = data.data.features
        self.target = data.data.targets

        target_column_name = self.target.columns[0]
        data = pd.concat([self.df, self.target], axis=1)
        data = data.dropna(subset=[target_column_name])

        self.df = data.iloc[:, :-1]
        self.target = data.iloc[:, -1]  # All rows, the last column
        super().__init__(train_size=160, *args, **kwargs)

class Segmentaition(Test):
    def __init__(self, *args, **kwargs) -> None:

        file_path = 'data/segmentation/segmentation.data'
        data = pd.read_csv(file_path)


        self.df = data.iloc[:, :-1]
        self.target = "class"
        super().__init__(train_size=1500, *args, **kwargs)


class Wine(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/wine/wine.data"
        self.df = pd.read_csv(path, header=None)
        self.target = 0
        super().__init__(train_size=124, *args, **kwargs)


class Thyroid(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/thyroid/new-thyroid.data"
        self.df = pd.read_csv(path, header=None)
        self.target = 0
        super().__init__(train_size=150, *args, **kwargs)


class Immunotherapy(Test):
    def __init__(self, *args, **kwargs) -> None:

        file_path = './data/immunotherapy/Immunotherapy.xlsx'
        data = pd.read_excel(file_path)

        self.df = data.drop('Result_of_Treatment', axis=1)
        self.target = data['Result_of_Treatment']

        super().__init__(train_size=63  , *args, **kwargs)

class Autism(Test):
    def __init__(self, *args, **kwargs) -> None:

        file_path = './data/Autism/Toddler Autism dataset July 2018.csv'
        self.df = pd.read_csv(file_path).drop('Case_No', axis=1)

        self.target = 'Class/ASD Traits '

        super().__init__(train_size=737, *args, **kwargs)


class Iris(Test):
    def __init__(self, *args, **kwargs) -> None:
        data = datasets.load_iris(as_frame=True)
        self.target = data.target
        self.df = data.data
        super().__init__(train_size=105, *args, **kwargs)
