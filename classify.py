# -*- coding: utf-8 -*-
import numpy as np
from model import *
from loader import Loader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from statistics import mean, median, variance, stdev
import keras.backend.tensorflow_backend as KTF


def classify(mode_auto, mode_classify, n_class, train_rate):
    if mode_auto == "AE":
        dist = np.load("distributed/{}.npy".format(mode_auto))
    elif mode_auto == "CAE":
        dist = np.load("distributed/{}.npy".format(mode_auto))
    else:
        raise Exception
    print("学習用データを読み取りました.")
    print("サイズは{}です".format(dist.shape))

    label = []
    # ラベル作成．全クラスの個数が同じという仮定をおいているので違う場合はよしなに変更する
    for i in range(n_class):
        for j in range(dist.shape[0] // n_class):
            label.append(i)

    x_train, y_train, x_test, y_test = dist[:int(len(dist)*train_rate)], label[:int(len(dist)*train_rate)], dist[int(len(dist)*train_rate):], label[int(len(dist)*train_rate):]
    x_train, x_test = x_train.tolist(), x_test.tolist()

    if mode_classify == "SVM":
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        }
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=8)

        grid_search.fit(x_train, y_train)

        print('Best parameters: {}'.format(grid_search.best_params_))
        print('Best cross-validation: {}'.format(grid_search.best_score_))

        clf = SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], kernel="rbf")
        clf.fit(x_train, y_train)

        scores = cross_val_score(clf, x_train, y_train)
        print('Cross-Validation accuracy: {}'.format(scores))
        print('Average accuracy: {}'.format(np.mean(scores)))

        train_accuracy = np.mean(scores)

        y_pred_test = clf.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print("test accuracy", test_accuracy)

        y_pred_test = y_pred_test.tolist()

        precision, recall, f, support = precision_recall_fscore_support(y_test, y_pred_test, average=None)
        print(confusion_matrix(y_test, y_pred_test, labels=list(range(n_class))))

    elif mode_classify == "RF":
        parameters = {
            "n_estimators": [i for i in range(10, 100, 10)],
            "criterion": ["gini", "entropy"],
            "max_depth": [i for i in range(1, 6)],
            'min_samples_split': [2, 4, 10, 12, 16],
        }
        grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=2, n_jobs=8)
        grid_search.fit(x_train, y_train)

        print('Best parameters: {}'.format(grid_search.best_params_))
        print('Best cross-validation: {}'.format(grid_search.best_score_))
        model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], criterion=grid_search.best_params_['criterion'],
                                       max_depth=grid_search.best_params_['max_depth'], min_samples_split=grid_search.best_params_['min_samples_split'])
        model.fit(x_train, y_train)
        importances = model.feature_importances_
        # print(max(importances))
        # print(np.argmax(importances))
        scores = cross_val_score(model, x_train, y_train)
        print('Cross-Validation accuracy: {}'.format(scores))

        train_accuracy = np.mean(scores)
        y_pred_test = model.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print("train accuracy", train_accuracy)
        print("test accuracy", test_accuracy)

        y_pred_test = y_pred_test.tolist()

        precision, recall, f, support = precision_recall_fscore_support(y_test, y_pred_test, average=None)
        print(confusion_matrix(y_test, y_pred_test, labels=list(range(n_class))))

    else:
        raise Exception


if __name__ == "__main__":
    # mode_auto = "AE"
    mode_auto = "CAE"

    mode_classify = "RF"
    # mode_classify = "SVM"

    n_class = 2
    train_rate = 0.8

    classify(mode_auto=mode_auto,
             mode_classify=mode_classify,
             n_class=n_class,
             train_rate=train_rate
             )
