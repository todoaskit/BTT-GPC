from collections import defaultdict
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from dataset import DataLoader, FILE_NAME

"""
Reference
http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc.html
http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html
"""


class GPC:

    def __init__(self, kernel_list, n_splits=5):
        # Data
        self.loader = DataLoader('{}.txt'.format(FILE_NAME))

        # Model
        self.gpc_dict: Dict[int, List[GaussianProcessClassifier]] = defaultdict(list)
        for k in range(n_splits):
            for kernel in kernel_list:
                self.gpc_dict[k].append(GaussianProcessClassifier(kernel=kernel))

        print('initialized with {} kernels, {} n_splits'.format(len(kernel_list), n_splits))

    def fit(self, fold):
        X_train, y_train, X_test, y_test = self.loader.get_train_test_xy(fold)
        for gpc in self.gpc_dict[fold]:
            gpc.fit(X_train, y_train)
            print('fit: {}'.format(gpc.kernel))

    def eval(self, fold):
        X_train, y_train, X_test, y_test = self.loader.get_train_test_xy(fold)
        for gpc in self.gpc_dict[fold]:
            train_acc = accuracy_score(y_train, gpc.predict(X_train))
            test_acc = accuracy_score(y_test, gpc.predict(X_test))
            print("="*10)
            print("Kernel: {}".format(gpc.kernel))
            print("Train Acc: {}".format(train_acc))
            print("Test Acc: {}".format(test_acc))


if __name__ == '__main__':
    gp_classifier = GPC(
        kernel_list=[
            1.0 * RBF([1.0]),
        ],
    )
    gp_classifier.fit(fold=0)
    gp_classifier.eval(fold=0)
