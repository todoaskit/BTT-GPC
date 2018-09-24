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

if __name__ == '__main__':

    # Data
    FOLD = 0
    loader = DataLoader('{}.txt'.format(FILE_NAME))
    X_train, y_train, X_test, y_test = loader.get_train_test_xy(fold=FOLD)

    # Model
    kernel = 1.0 * RBF([1.0])
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, gpc.predict(X_train))
    test_acc = accuracy_score(y_test, gpc.predict(X_test))

    print("Train Accuracy: {}".format(train_acc))
    print("Test Accuracy: {}".format(test_acc))
