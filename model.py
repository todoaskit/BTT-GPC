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

    def __init__(self, kernel, loader: DataLoader = None, n_splits: int = 5):
        # Data
        self.loader = loader or DataLoader('{}.txt'.format(FILE_NAME))
        if not loader:
            self.loader.transform_x(manifold.TSNE, n_components=3)

        # Model
        self.gpc_dict: Dict[int, GaussianProcessClassifier] = dict()
        for k in range(n_splits):
            self.gpc_dict[k] = GaussianProcessClassifier(kernel=deepcopy(kernel))

        print('initialized with {}, n_splits: {}'.format(kernel, n_splits))

    @timeit
    def fit(self, fold):
        X_train, y_train, X_test, y_test = self.loader.get_train_test_xy(fold)
        self.gpc_dict[fold].fit(X_train, y_train)
        print('fit: {}'.format(self.gpc_dict[fold].kernel))

    @timeit
    def eval(self, fold):
        X_train, y_train, X_test, y_test = self.loader.get_train_test_xy(fold)
        gpc = self.gpc_dict[fold]
        train_acc = accuracy_score(y_train, gpc.predict(X_train))
        test_acc = accuracy_score(y_test, gpc.predict(X_test))
        print("Fold: {}, Kernel: {}".format(fold, gpc.kernel))
        print("Train Acc: {}".format(train_acc))
        print("Test Acc: {}".format(test_acc))
        print("=" * 10)

    def run(self, fold):
        self.fit(fold)
        self.eval(fold)

    def run_all(self):
        for fold in range(len(self.gpc_dict)):
            self.run(fold)


if __name__ == '__main__':

    now_fold = 0

    data_loader = DataLoader('{}.txt'.format(FILE_NAME))
    data_loader.transform_x(manifold.TSNE, n_components=3)

    kernels = [
        1.0 * Matern(length_scale=random() * randrange(1, 5),
                     length_scale_bounds=(1e-5, 1e5),
                     nu=2.5)
        + 1.0 * Matern(length_scale=random() * randrange(1, 5),
                       length_scale_bounds=(1e-5, 1e5),
                       nu=1.5)
    ]

    for now_kernel in kernels:
        try:
            gp_classifier = GPC(
                kernel=now_kernel,
                loader=data_loader,
            )
            gp_classifier.run(fold=now_fold)
        except Exception as e:
            print(e)
