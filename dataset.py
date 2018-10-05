import csv
import itertools
import pickle
import random
from typing import Callable

import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import decomposition
from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler

from useful import timeit

FILE_NAME = 'Brain_gene_exp.rm_donor_specific'


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

    def get_n_splits(self):
        return len(self.kfold_split)

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
        scaler = MinMaxScaler()
        scaler.fit(self.x_data)
        self.x_data = scaler.transform(self.x_data)
        print('Transform: {}, {}'.format(transform_func.__name__, kwargs))

    def get_colors(self, pal='hls'):
        y_to_color = sns.color_palette(pal, len(self.y_embedding)).as_hex()
        colors = np.empty(0, dtype='float')
        for y in self.y_data:
            colors = np.append(colors, y_to_color[y])
        return colors

    def display_2d(self, transform_func: Callable, **kwargs):
        colors = self.get_colors()
        x_data_2d = self.get_transform_x(transform_func, n_components=2, **kwargs)
        x_coord = x_data_2d[:, 0]
        y_coord = x_data_2d[:, 1]

        plt.scatter(x_coord, y_coord, c=colors, s=10)
        plt.xlim(x_coord.min() + 0.00005, x_coord.max() + 0.00005)
        plt.ylim(y_coord.min() + 0.00005, y_coord.max() + 0.00005)
        plt.show()

    def display_3d(self, transform_func: Callable, **kwargs):
        colors = self.get_colors()
        x_data_3d = self.get_transform_x(transform_func, n_components=3, **kwargs)
        x_coord = x_data_3d[:, 0]
        y_coord = x_data_3d[:, 1]
        z_coord = x_data_3d[:, 2]

        x_mean, y_mean, z_mean = map(np.mean, [x_coord, y_coord, z_coord])
        x_std, y_std, z_std = map(np.std, [x_coord, y_coord, z_coord])

        ax = plt.axes(projection='3d')
        ax.scatter(x_coord, y_coord, z_coord, c=colors, s=10)
        ax.set_xlim3d(x_mean - 1.3 * x_std, x_mean + 1.3 * x_std)
        ax.set_ylim3d(y_mean - 1.3 * y_std, y_mean + 1.3 * y_std)
        ax.set_zlim3d(z_mean - 1.3 * z_std, z_mean + 1.3 * z_std)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


if __name__ == '__main__':
    loader = DataLoader('{}.tsv'.format(FILE_NAME))
    loader.transform_x(decomposition.PCA, n_components=80)
    loader.display_2d(manifold.TSNE, metric="correlation", random_state=21)
    loader.display_2d(manifold.TSNE, metric="cosine", random_state=21)
    loader.display_2d(manifold.TSNE, random_state=21)
