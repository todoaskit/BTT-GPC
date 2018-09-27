import csv
import itertools
import pickle
import random
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import decomposition
from sklearn import manifold

from useful import timeit

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

    def get_transform_x(self, transform_func: Callable, **kwargs):
        trans = transform_func(**kwargs)
        transformed_x_data = trans.fit_transform(self.x_data)
        return transformed_x_data

    @timeit
    def transform_x(self, transform_func: Callable, **kwargs):
        self.x_data = self.get_transform_x(transform_func, **kwargs)
        print('Transform: {}, {}'.format(transform_func.__name__, kwargs))

    def display_2d(self, transform_func: Callable):
        x_data_2d = self.get_transform_x(transform_func, n_components=2, random_state=13)

        random_color = lambda: "#%06x" % random.randint(0, 0xFFFFFF)
        y_to_color = {label: random_color() for label in self.y_embedding.values()}
        colors = np.empty(0, dtype='float')
        for y in self.y_data:
            colors = np.append(colors, y_to_color[y])

        x_coord = x_data_2d[:, 0]
        y_coord = x_data_2d[:, 1]

        plt.scatter(x_coord, y_coord, c=colors, s=10)
        plt.xlim(x_coord.min() + 0.00005, x_coord.max() + 0.00005)
        plt.ylim(y_coord.min() + 0.00005, y_coord.max() + 0.00005)
        plt.show()


if __name__ == '__main__':
    loader = DataLoader('{}.txt'.format(FILE_NAME))
    loader.display_2d(manifold.TSNE)
