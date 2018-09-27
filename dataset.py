import csv
import itertools
import pickle
from typing import Callable

import numpy as np
from sklearn.model_selection import KFold


FILE_NAME = 'TissuePrediction_train'


class DataLoader:

    def __init__(self, file_name, n_splits=5, shuffle=True):
        """
        Attribute
        ---------
        x_label: list
        x_data: list of np.ndarray
        y_data: list of int
        y_embedding: dict from str to int
        """
        self.file_name = file_name
        self.x_label, self.x_data, self.y_data, self.y_embedding = None, None, None, None

        if not self.load():
            self.x_label, self.x_data, self.y_data, self.y_embedding = self.load_txt()
            self.dump()

        cv = KFold(n_splits=n_splits, shuffle=shuffle)
        self.kfold_split = list(cv.split(self.x_data))

    def dump(self, dump_name='./{}.pkl'.format(FILE_NAME)):
        with open(dump_name, 'wb') as f:
            pickle.dump(self, f)
            print('Dumped: {}'.format(dump_name))

    def load(self, load_name='./{}.pkl'.format(FILE_NAME)):
        try:
            with open(load_name, 'rb') as f:
                loaded: DataLoader = pickle.load(f)
                self.file_name = loaded.file_name
                self.x_label = loaded.x_label
                self.x_data = loaded.x_data
                self.y_data = loaded.y_data
                self.y_embedding = loaded.y_embedding
            print('Loaded: {}'.format(load_name))
            return True
        except Exception as e:
            print(e)
            return False

    def load_txt(self):
        f = open(self.file_name, 'r')
        all_data = list(csv.reader(f, delimiter='\t'))
        trans_data = [list(x) for x in itertools.zip_longest(*all_data, fillvalue='')]

        # experiment, tissue, donor, x ...
        x_label = trans_data[0][3:]
        x_data = [np.asarray([float(x) for x in yx[3:]], dtype=np.float32) for yx in trans_data[1:]]

        y_data = [yx[1] for yx in trans_data[1:]]
        y_embedding = {y: i for i, y in enumerate(sorted(set(y_data)))}
        y_data_embed = [y_embedding[y] for y in y_data]

        return x_label, x_data, y_data_embed, y_embedding

    def __len__(self):
        assert len(self.x_data) == len(self.y_data)
        return len(self.x_data)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def get_x_label(self, item):
        return self.x_label[item]

    def get_train_test_xy(self, fold):
        X = np.asarray(self.x_data, dtype=np.float32)
        y = np.asarray(self.y_data, dtype=int)
        train_idx, test_idx = self.kfold_split[fold]
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def decomposition_x(self, decomposition_func: Callable, **kwargs):
        decmp = decomposition_func(**kwargs)
        decmp.fit(self.x_data)
        self.x_data = decmp.transform(self.x_data)
        print('Data decomposition: {}, {}'.format(decomposition_func.__name__, kwargs))


if __name__ == '__main__':
    loader = DataLoader('{}.txt'.format(FILE_NAME))
    X_train, y_train, X_test, y_test = loader.get_train_test_xy(0)
    print(X_train, len(X_train))
    print(y_train, len(y_train))
    print(X_test, len(X_test))
    print(y_test, len(y_test))
