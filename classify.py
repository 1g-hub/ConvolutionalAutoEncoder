# -*- coding: utf-8 -*-
import numpy as np
from utils.model import *
from utils.loader import Loader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys
from statistics import mean, median, variance, stdev


def classify(mode_auto, mode_classify, save_dir, n_class, load_dir, dic_label, acc_dir, pred_dir, metrics_dir, experiment, weight_all=False):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    old_session = KTF.get_session()

    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)

    acc_df = pd.read_csv("{}/accuracy.csv".format(acc_dir), encoding="utf-8", index_col=0)
    pred_df = pd.read_csv("{}/predict.csv".format(pred_dir), encoding="utf-8", index_col=0,
                          dtype={"index": "int", "img": "str", "label": "str", "AE_MLP_pred": "str", "AE_MLP_ok": "bool",
                                 "CAE_MLP_pred": "str", "CAE_MLP_ok": "bool", "simple_cnn_pred": "str", "simple_cnn_ok": "bool",
                                 "AE_SVM_pred": "str", "AE_SVM_ok": "bool", "CAE_SVM_pred": "str", "CAE_SVM_ok": "bool",
                                 "AE_RF_pred": "str", "AE_RF_ok": "bool", "CAE_RF_pred": "str", "CAE_RF_ok": "bool"})
    metrics_df = pd.read_csv("{}/metrics.csv".format(metrics_dir), encoding="utf-8", index_col=0)

    label = []

    if mode_classify == "mlp_with_dist":
        if mode_auto == "AE":
            if weight_all:
                dist = np.load("distributed/exp{}/{}_weight_all.npy".format(experiment, mode_auto))
            else:
                dist = np.load("distributed/exp{}/{}.npy".format(experiment, mode_auto))
            print("学習用データを読み取りました.")
        elif mode_auto == "CAE":
            if weight_all:
                dist = np.load("distributed/exp{}/{}_weight_all.npy".format(experiment, mode_auto))
            else:
                dist = np.load("distributed/exp{}/{}.npy".format(experiment, mode_auto))
            print("学習用データを読み取りました.")
        else:
            raise Exception

        input_shape = ((dist.shape[1]),)

        if experiment == "1_4":
            for _ in range(64):  # 萌えタッチの数
                label.append(0)
            for _ in range(74):  # 青年漫画タッチの数
                label.append(1)
            for _ in range(66):  # 少年漫画タッチの数
                label.append(2)
        elif experiment == "1_5":
            for _ in range(63):  # 萌えタッチの数
                label.append(0)
            for _ in range(64):  # 青年漫画タッチの数
                label.append(1)
            for _ in range(69):  # 少年漫画タッチの数
                label.append(2)
        else:
            for i in range(n_class):
                for j in range(dist.shape[0] // n_class):
                    label.append(i)

        label = np_utils.to_categorical(np.array(label))

        train_accuracy, test_accuracy, test_img, y_test, y_pred_test = train_classify(mode_classify=mode_classify, mode_auto=mode_auto, data=dist, label=label, n_class=n_class, input_shape=input_shape, batch_size=32, verbose=1, epochs=200,
                  validation_split=0.2, show=True, saveDir=save_dir, experiment=experiment)
        y_test = [np.argmax(y_test[i]) for i in range(len(y_test))]
        precision, recall, f, support = precision_recall_fscore_support(y_test, y_pred_test, average=None)
        y_pred_test = y_pred_test.tolist()
        print(confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2]))
        if mode_auto == "AE":
            for i in range(len(dist)):
                if i not in test_img:
                    pred_df.at[i, "AE_MLP_pred"] = None
                    pred_df.at[i, "AE_MLP_ok"] = None
                else:
                    tmp = dic_label[y_pred_test.pop(0)]
                    pred_df.at[i, "AE_MLP_pred"] = tmp
                    if tmp == pred_df.at[i, "label"]:
                        pred_df.at[i, "AE_MLP_ok"] = True
                    else:
                        pred_df.at[i, "AE_MLP_ok"] = False

            acc_df.at["train", "AE_MLP"] = train_accuracy
            acc_df.at["test", "AE_MLP"] = test_accuracy
            metrics_df["AE_MLP_moe"] = [precision[0], recall[0], f[0]]
            metrics_df["AE_MLP_seinen"] = [precision[1], recall[1], f[1]]
            metrics_df["AE_MLP_shonen"] = [precision[2], recall[2], f[2]]

        else:
            for i in range(len(dist)):
                if i not in test_img:
                    pred_df.at[i, "CAE_MLP_pred"] = None
                    pred_df.at[i, "CAE_MLP_ok"] = None
                else:
                    tmp = dic_label[y_pred_test.pop(0)]
                    pred_df.at[i, "CAE_MLP_pred"] = tmp
                    if tmp == pred_df.at[i, "label"]:
                        pred_df.at[i, "CAE_MLP_ok"] = True
                    else:
                        pred_df.at[i, "CAE_MLP_ok"] = False

            acc_df.at["train", "CAE_MLP"] = train_accuracy
            acc_df.at["test", "CAE_MLP"] = test_accuracy
            metrics_df["CAE_MLP_moe"] = [precision[0], recall[0], f[0]]
            metrics_df["CAE_MLP_seinen"] = [precision[1], recall[1], f[1]]
            metrics_df["CAE_MLP_shonen"] = [precision[2], recall[2], f[2]]

    elif mode_classify == "simple_cnn":
        loader_ins = Loader(load_dir)
        loader_ins.load(gray=True, size=(196, 136))  # 横×縦
        print("学習用データを読み取りました.")
        data = loader_ins.get_data(norm=True)
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))

        if experiment == "1_4":
            for _ in range(64):  # 萌えタッチの数
                label.append(0)
            for _ in range(74):  # 青年漫画タッチの数
                label.append(1)
            for _ in range(66):  # 少年漫画タッチの数
                label.append(2)
        elif experiment == "1_5":
            for _ in range(63):  # 萌えタッチの数
                label.append(0)
            for _ in range(64):  # 青年漫画タッチの数
                label.append(1)
            for _ in range(69):  # 少年漫画タッチの数
                label.append(2)
        else:
            for i in range(n_class):
                for j in range(data.shape[0] // n_class):
                    label.append(i)

        label = np_utils.to_categorical(np.array(label))

        input_shape = (data.shape[1], data.shape[2], 1)
        train_accuracy, test_accuracy, test_img, y_test, y_pred_test = train_classify(mode_classify=mode_classify, mode_auto=None, data=data, label=label, n_class=n_class, input_shape=input_shape, batch_size=64, verbose=1, epochs=200,
                  validation_split=0.2, show=True, saveDir=save_dir, experiment=experiment)

        y_test = [np.argmax(y_test[i]) for i in range(len(y_test))]
        y_pred_test = y_pred_test.tolist()

        precision, recall, f, support = precision_recall_fscore_support(y_test, y_pred_test, average=None)
        print(confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2]))
        for i in range(len(data)):
            if i not in test_img:
                pred_df.at[i, "simple_cnn_pred"] = None
                pred_df.at[i, "simple_cnn_ok"] = None
            else:
                tmp = dic_label[y_pred_test.pop(0)]
                pred_df.at[i, "simple_cnn_pred"] = tmp
                if tmp == pred_df.at[i, "label"]:
                    pred_df.at[i, "simple_cnn_ok"] = True
                else:
                    pred_df.at[i, "simple_cnn_ok"] = False

        acc_df.at["train", "simple_cnn"] = train_accuracy
        acc_df.at["test", "simple_cnn"] = test_accuracy
        metrics_df["simple_cnn_moe"] = [precision[0], recall[0], f[0]]
        metrics_df["simple_cnn_seinen"] = [precision[1], recall[1], f[1]]
        metrics_df["simple_cnn_shonen"] = [precision[2], recall[2], f[2]]

    elif mode_classify == "SVM" or mode_classify == "RF":
        if mode_auto == "AE":
            if weight_all:
                dist = np.load("distributed/exp{}/{}_weight_all.npy".format(experiment, mode_auto))
                print("重みをALLにした学習用データを読み取りました.")
            else:
                dist = np.load("distributed/exp{}/{}.npy".format(experiment, mode_auto))
                print("学習用データを読み取りました.")
        elif mode_auto == "CAE":
            if weight_all:
                dist = np.load("distributed/exp{}/{}_weight_all.npy".format(experiment, mode_auto))
                print("重みをALLにした学習用データを読み取りました.")
            else:
                dist = np.load("distributed/exp{}/{}.npy".format(experiment, mode_auto))
                print("学習用データを読み取りました.")
        else:
            raise Exception

        if experiment == "1_4":
            for _ in range(64):  # 萌えタッチの数
                label.append(0)
            for _ in range(74):  # 青年漫画タッチの数
                label.append(1)
            for _ in range(66):  # 少年漫画タッチの数
                label.append(2)
        elif experiment == "1_5":
            for _ in range(63):  # 萌えタッチの数
                label.append(0)
            for _ in range(64):  # 青年漫画タッチの数
                label.append(1)
            for _ in range(69):  # 少年漫画タッチの数
                label.append(2)
        else:
            for i in range(n_class):
                for j in range(dist.shape[0] // n_class):
                    label.append(i)

        (x_train, y_train), (x_test, y_test), test_img = train_test_split(data=dist, label=label, n_class=n_class, train_rate=0.8)
        x_train, y_train, x_test, y_test = x_train.tolist(), y_train.tolist(), x_test.tolist(), y_test.tolist()

        if mode_classify == "SVM":
            param_grid = {
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'kernel': ['rbf', 'linear', 'poly'],
                'random_state': [1, 2, 3]
            }
            grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=8)

            grid_search.fit(x_train, y_train)

            print('Best parameters: {}'.format(grid_search.best_params_))
            print('Best cross-validation: {}'.format(grid_search.best_score_))

            clf = SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'],
                      kernel=grid_search.best_params_['kernel'], random_state=grid_search.best_params_['random_state'])
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
            print(confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2]))
            for i in range(len(dist)):
                if i not in test_img:
                    pred_df.at[i, "{}_SVM_pred".format(mode_auto)] = None
                    pred_df.at[i, "{}_SVM_ok".format(mode_auto)] = None
                else:
                    tmp = dic_label[y_pred_test.pop(0)]
                    pred_df.at[i, "{}_SVM_pred".format(mode_auto)] = tmp
                    if tmp == pred_df.at[i, "label"]:
                        pred_df.at[i, "{}_SVM_ok".format(mode_auto)] = True
                    else:
                        pred_df.at[i, "{}_SVM_ok".format(mode_auto)] = False

            acc_df.at["train", "{}_SVM".format(mode_auto)] = train_accuracy
            acc_df.at["test", "{}_SVM".format(mode_auto)] = test_accuracy
            metrics_df["{}_SVM_moe".format(mode_auto)] = [precision[0], recall[0], f[0]]
            metrics_df["{}_SVM_seinen".format(mode_auto)] = [precision[1], recall[1], f[1]]
            metrics_df["{}_SVM_shonen".format(mode_auto)] = [precision[2], recall[2], f[2]]

        elif mode_classify == "RF":
            parameters = {
                "n_estimators": [i for i in range(10, 100, 10)],
                "criterion": ["gini", "entropy"],
                "max_depth": [i for i in range(1, 6, 1)],
                'min_samples_split': [2, 4, 10, 12, 16],
                "random_state": [1, 2, 3]
            }
            grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1)
            grid_search.fit(x_train, y_train)

            print('Best parameters: {}'.format(grid_search.best_params_))
            print('Best cross-validation: {}'.format(grid_search.best_score_))
            model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], criterion=grid_search.best_params_['criterion'],
                                           max_depth=grid_search.best_params_['max_depth'], min_samples_split=grid_search.best_params_['min_samples_split'],
                                           random_state=grid_search.best_params_['random_state'])
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
            print(confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2]))
            with open("results/exp{}/confusion_matrix.txt".format(experiment), "w") as file:
                mat = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])
                file.write("{} {} {}\n".format(str(mat[0][0]), str(mat[0][1]), str(mat[0][2])))
                file.write("{} {} {}\n".format(str(mat[1][0]), str(mat[1][1]), str(mat[1][2])))
                file.write("{} {} {}\n".format(str(mat[2][0]), str(mat[2][1]), str(mat[2][2])))
            for i in range(len(dist)):
                if i not in test_img:
                    pred_df.at[i, "{}_RF_pred".format(mode_auto)] = None
                    pred_df.at[i, "{}_RF_ok".format(mode_auto)] = None
                else:
                    tmp = dic_label[y_pred_test.pop(0)]
                    pred_df.at[i, "{}_RF_pred".format(mode_auto)] = tmp
                    if tmp == pred_df.at[i, "label"]:
                        pred_df.at[i, "{}_RF_ok".format(mode_auto)] = True
                    else:
                        pred_df.at[i, "{}_RF_ok".format(mode_auto)] = False

            acc_df.at["train", "{}_RF".format(mode_auto)] = train_accuracy
            acc_df.at["test", "{}_RF".format(mode_auto)] = test_accuracy
            metrics_df["{}_RF_moe".format(mode_auto)] = [precision[0], recall[0], f[0]]
            metrics_df["{}_RF_seinen".format(mode_auto)] = [precision[1], recall[1], f[1]]
            metrics_df["{}_RF_shonen".format(mode_auto)] = [precision[2], recall[2], f[2]]

        else:
            raise Exception
    else:
        raise Exception

    pred_df.to_csv("{}/predict.csv".format(pred_dir))
    print("predictを保存しました．")
    acc_df.to_csv("{}/accuracy.csv".format(acc_dir))
    print("accuracyを保存しました．")
    metrics_df.to_csv("{}/metrics.csv".format(metrics_dir))
    print("metricsを保存しました．")

    KTF.set_session(old_session)

    return test_accuracy


if __name__ == "__main__":
    save_dir = os.getcwd()

    # mode_auto = "AE"
    mode_auto = "CAE"

    # mode_classify = "mlp_with_dist"
    mode_classify = "RF"
    # mode_classify = "SVM"

    # mode_classify = "simple_cnn"

    """
    experiment=1_1:設定ALL
    experiment=1_2:設定REMOVE_EYE
    experiment=1_3:設定REMOVE_MOUTH
    experiment=1_4:設定EYE_ONLY
    experiment=1_5:設定MOUTH_ONLY
    """
    experiment = "1_2"

    dic_label = {0: "moe", 1: "seinen", 2: "shonen"}
    n_class = 3

    if experiment == "1_1":
        load_dir = os.path.join(os.getcwd(), "imgs_param/ALL")
    elif experiment == "1_2":
        load_dir = os.path.join(os.getcwd(), "imgs_param/REMOVE_EYE")
    elif experiment == "1_3":
        load_dir = os.path.join(os.getcwd(), "imgs_param/REMOVE_MOUTH")
    elif experiment == "1_4":
        load_dir = os.path.join(os.getcwd(), "imgs_param/EYE_ONLY")
    elif experiment == "1_5":
        load_dir = os.path.join(os.getcwd(), "imgs_param/MOUTH_ONLY")
    else:
        pass
    weight_all = False
    acc_dir = "results/exp{}".format(experiment)
    pred_dir = "results/exp{}".format(experiment)
    metrics_dir = "results/exp{}".format(experiment)

    test_acc_list = []
    N = 10
    for i in range(N):
        test_acc_list.append(classify(mode_auto=mode_auto, mode_classify=mode_classify, save_dir=save_dir, n_class=n_class, load_dir=load_dir, dic_label=dic_label, acc_dir=acc_dir, pred_dir=pred_dir, metrics_dir=metrics_dir, experiment=experiment, weight_all=weight_all))

    print("実験：{}".format(experiment))
    print("平均：{}".format(mean(test_acc_list)))
    print("標準偏差：{}".format(stdev(test_acc_list)))
