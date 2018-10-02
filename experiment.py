import numpy as np
from sklearn import decomposition

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV

from dataset import DataLoader, FILE_NAME


def run_experiment_all_folds(clf_cls, loader, print_result=True, **kwargs):
    train_acc_list, test_acc_list = [], []
    for fold in range(loader.get_n_splits()):
        train_acc, test_acc = run_experiment(clf_cls, loader, fold, print_result=False, **kwargs)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    train_acc_avg = np.average(train_acc_list)
    train_acc_stdev = np.std(train_acc_list)
    test_acc_avg = np.average(test_acc_list)
    test_acc_stdev = np.std(test_acc_list)

    if print_result:
        print("{}, K-fold: {}, params: {}".format(clf_cls.__name__, loader.get_n_splits(), kwargs))
        print("Train Acc: {} (+/- {})".format(train_acc_avg, train_acc_stdev))
        print("Test Acc: {} (+/- {})".format(test_acc_avg, test_acc_stdev))
        print("=" * 10)

    return train_acc_list, test_acc_list


def run_experiment(clf_cls, loader, fold, print_result=True, **kwargs):
    X_train, y_train, X_test, y_test = loader.get_train_test_xy(fold)
    clf = clf_cls(**kwargs)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    if print_result:
        print("{}, fold: {}, params: {}".format(clf_cls.__name__, fold, kwargs))
        print("Train Acc: {}".format(train_acc))
        print("Test Acc: {}".format(test_acc))
        print("=" * 10)

    return train_acc, test_acc


def print_grid_search(clf_instance, loader, params_to_search, cv=2):
    clf = GridSearchCV(clf_instance, params_to_search, cv=cv)
    clf.fit(loader.x_data, loader.y_data)
    print('Best: {}'.format(clf.best_params_))
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("{} (+/- {}) for {}".format(mean, std * 2, params))


if __name__ == '__main__':

    data_loader = DataLoader('{}.txt'.format(FILE_NAME))
    data_loader.transform_x(decomposition.PCA, n_components=80)

    run_experiment_all_folds(QuadraticDiscriminantAnalysis, data_loader)
    run_experiment_all_folds(AdaBoostClassifier, data_loader)
    run_experiment_all_folds(GaussianNB, data_loader)
    run_experiment_all_folds(MLPClassifier, data_loader, alpha=1)
    run_experiment_all_folds(KNeighborsClassifier, data_loader, n_neighbors=5)
    run_experiment_all_folds(DecisionTreeClassifier, data_loader, max_depth=5)
    run_experiment_all_folds(RandomForestClassifier, data_loader, n_estimators=2500, max_depth=5)
    run_experiment_all_folds(SVC, data_loader, kernel="linear", C=0.025)
    run_experiment_all_folds(SVC, data_loader, gamma=2, C=1)
