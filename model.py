from copy import deepcopy
from random import randrange, random
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern
from sklearn import decomposition
from sklearn import manifold

from dataset import DataLoader, FILE_NAME
from useful import timeit

"""
References for GPC
http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc.html
http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html

For kernels,
http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
"""


class GPC:

    def __init__(self, kernel, loader: DataLoader = None, n_splits: int = 5, **kwargs):
        # Data
        self.loader = loader or DataLoader('{}.txt'.format(FILE_NAME), n_splits=n_splits)
        if not loader:
            self.loader.transform_x(manifold.TSNE, n_components=3)

        # Model
        self.gpc_dict: Dict[int, GaussianProcessClassifier] = dict()
        for k in range(loader.get_n_splits()):
            self.gpc_dict[k] = GaussianProcessClassifier(kernel=deepcopy(kernel), **kwargs)

        print('initialized with {}, n_splits: {}'.format(kernel, loader.get_n_splits()))

    @timeit
    def fit(self, fold):
        assert fold < self.loader.get_n_splits(), "fold >= {}".format(self.loader.get_n_splits())
        X_train, y_train, X_test, y_test = self.loader.get_train_test_xy(fold)
        self.gpc_dict[fold].fit(X_train, y_train)
        print('fit: {}'.format(self.gpc_dict[fold].kernel))

    @timeit
    def eval(self, fold, print_result=True):
        assert fold < self.loader.get_n_splits(), "fold >= {}".format(self.loader.get_n_splits())
        X_train, y_train, X_test, y_test = self.loader.get_train_test_xy(fold)
        gpc = self.gpc_dict[fold]
        train_acc = accuracy_score(y_train, gpc.predict(X_train))
        test_acc = accuracy_score(y_test, gpc.predict(X_test))

        if print_result:
            print("Fold: {}, Kernel: {}".format(fold, gpc.kernel))
            print("Train Acc: {}".format(train_acc))
            print("Test Acc: {}".format(test_acc))
            print("=" * 10)

        return train_acc, test_acc

    def run(self, fold, print_result=True):
        self.fit(fold)
        return self.eval(fold, print_result=print_result)

    def run_all(self, print_result=True):
        train_acc_list, test_acc_list = [], []
        for fold in range(self.loader.get_n_splits()):
            train_acc, test_acc = self.run(fold, print_result=False)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

        train_acc_avg = np.average(train_acc_list)
        train_acc_stdev = np.std(train_acc_list)
        test_acc_avg = np.average(test_acc_list)
        test_acc_stdev = np.std(test_acc_list)

        if print_result:
            print("K-fold: {}, Kernel: {}".format(self.loader.get_n_splits(), self.gpc_dict[0].kernel))
            print("Train Acc: {} (+/- {})".format(train_acc_avg, train_acc_stdev))
            print("Test Acc: {} (+/- {})".format(test_acc_avg, test_acc_stdev))
            print("=" * 10)

        return train_acc_list, test_acc_list


if __name__ == '__main__':

    now_fold = 0

    data_loader = DataLoader('{}.txt'.format(FILE_NAME))
    data_loader.transform_x(manifold.TSNE, n_components=3)

    now_kernel = 1.0 * Matern(length_scale=random() * randrange(1, 5),
                              length_scale_bounds=(1e-5, 1e5),
                              nu=1.5)

    try:
        gp_classifier = GPC(
            kernel=now_kernel,
            loader=data_loader,
            n_restarts_optimizer=2,
        )
        gp_classifier.run(fold=now_fold)
    except Exception as e:
        print(e)
