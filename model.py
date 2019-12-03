import numpy as np
import os
from keras import models
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.initializers import TruncatedNormal, Constant

from keras.layers import UpSampling2D, GlobalAveragePooling2D
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils
import matplotlib

from glob import glob
matplotlib.use("PS")
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model

from keras.callbacks import TensorBoard
import cv2 as cv
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import optuna


def get_latest_modified_file_path(model_dir):
  target = os.path.join(model_dir, '*')
  files = [(f, os.path.getmtime(f)) for f in glob(target)]
  latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
  return latest_modified_file_path[0]


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def model_mlp_classify(input_shape, n_class, mid_units, drop_rate):
    model = Sequential(
        [Dense(mid_units, input_shape=input_shape, activation="relu"), Dropout(drop_rate), Dense(units=n_class, activation="softmax")])
    print(model.summary())
    return model


def model_simple_cnn_classify(input_shape, n_class, num_layer, activation, mid_units, num_filters, drop_rate):
    model = Sequential()
    model.add(
        Conv2D(
            filters=num_filters[0],
            input_shape=input_shape,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=activation
        )
    )

    model.add(MaxPooling2D())
    for i in range(1, num_layer):
        model.add(Conv2D(
            filters=num_filters[i],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=activation
        ))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(units=mid_units, activation=activation))
    model.add(Dropout(drop_rate))
    model.add(Dense(units=n_class, activation='softmax'))

    print(model.summary())

    return model


def model_mlp_auto(input_shape):
    model = Sequential()
    model.add(Dense(units=5000, input_shape=input_shape, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(units=3332, input_shape=input_shape, activation="relu", name='mid'))
    model.add(Dropout(0.5))
    model.add(Dense(units=5000, input_shape=input_shape, activation="relu"))
    model.add(Dense(units=input_shape[0], activation="sigmoid"))
    print(model.summary())
    return model


def model_cnn_auto(input_shape):
    model = Sequential()
    model.add(
        Conv2D(
            filters=4,
            kernel_size=(4, 4),
            padding='same',
            activation='relu',
            input_shape=input_shape
        )
    )

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            filters=2,
            kernel_size=(4, 4),
            padding='same',
            activation='relu'
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='mid'))

    # ↑↑↑ここまでがEncoder

    # ↑↑↑ここからがDecoder
    model.add(
        Conv2D(
            filters=2,
            kernel_size=(4, 4),
            padding='same',
            activation='relu'
        )
    )
    model.add(UpSampling2D((2, 2)))

    model.add(
        Conv2D(
            filters=4,
            kernel_size=(4, 4),
            padding='same',
            activation='relu'
        )
    )

    # model.add(BatchNormalization())

    model.add(UpSampling2D((2, 2)))

    # model.add(Dropout(0.25))

    model.add(
        Conv2D(
            filters=1,
            kernel_size=(4, 4),
            padding='same',
            activation='sigmoid'
        )
    )
    print(model.summary())
    return model


# def model_cnn_auto(input_shape):  # filter の数をいじるよう
#     model = Sequential()
#     model.add(
#         Conv2D(
#             filters=256,
#             kernel_size=(4, 4),
#             padding='same',
#             activation='relu',
#             input_shape=input_shape
#         )
#     )
#
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(0.25))
#
#     model.add(
#         Conv2D(
#             filters=256,
#             kernel_size=(4, 4),
#             padding='same',
#             activation='relu'
#         )
#     )
#
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(0.25))
#
#     model.add(
#         Conv2D(
#             filters=128,
#             kernel_size=(4, 4),
#             padding='same',
#             activation='relu'
#         )
#     )
#
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(0.25))
#
#     model.add(
#         Conv2D(
#             filters=64,
#             kernel_size=(4, 4),
#             padding='same',
#             activation='relu'
#         )
#     )
#
#     model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
#
#     # ↑↑↑ここまでがEncoder
#
#     # ↑↑↑ここからがDecoder
#
#     model.add(
#         Conv2D(
#             filters=64,
#             kernel_size=(4, 4),
#             padding='same',
#             activation='relu'
#         )
#     )
#     model.add(UpSampling2D((2, 2)))
#
#     model.add(
#         Conv2D(
#             filters=128,
#             kernel_size=(4, 4),
#             padding='same',
#             activation='relu'
#         )
#     )
#     model.add(UpSampling2D((2, 2)))
#
#     model.add(
#         Conv2D(
#             filters=256,
#             kernel_size=(4, 4),
#             padding='same',
#             activation='relu'
#         )
#     )
#     model.add(UpSampling2D((2, 2)))
#
#     model.add(
#         Conv2D(
#             filters=256,
#             kernel_size=(4, 4),
#             padding='same',
#             activation='relu'
#         )
#     )
#
#     # model.add(BatchNormalization())
#
#     model.add(UpSampling2D((2, 2)))
#
#     # model.add(Dropout(0.25))
#
#     model.add(
#         Conv2D(
#             filters=1,
#             kernel_size=(4, 4),
#             padding='same',
#             activation='sigmoid'
#         )
#     )
#     print(model.summary())
#     return model


def train_val_split(data, n_class, train_rate=0.8):
    x_train = [data[i] for i in range(len(data)) if i % (len(data) // n_class) < (len(data) // n_class)*train_rate]
    x_val = [data[i] for i in range(len(data)) if i % (len(data) // n_class) >= (len(data) // n_class)*train_rate]
    x_train = np.array(x_train)
    x_val = np.array(x_val)

    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    return x_train, x_val


def train_auto(mode_auto, data, n_class, input_shape, saveDir, batch_size=100, verbose=1, epochs=20,
              validation_split=0.2, show=False, num_compare=5, experiment=None):
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        horizontal_flip=True,
        fill_mode='nearest')

    x_train, x_val = train_val_split(data, n_class)

    if mode_auto == "CAE":
        model = model_cnn_auto(input_shape)
    elif mode_auto == "AE":
        model = model_mlp_auto(input_shape)
    else:
        raise Exception
    model.compile(optimizer='Adam', loss='binary_crossentropy')

    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

    try:
        os.makedirs("tflog/exp{}/auto".format(experiment))
    except:
        pass
    if mode_auto == "CAE":
        tb_cb = TrainValTensorBoard(log_dir="tflog/exp{}/auto/{}".format(experiment, mode_auto), histogram_freq=1, write_graph=True)
        chkpt = os.path.join(saveDir, 'weights/auto/CAE/AutoEncoder_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
    elif mode_auto == "AE":
        tb_cb = TrainValTensorBoard(log_dir="tflog/exp{}/auto/{}".format(experiment, mode_auto), histogram_freq=1, write_graph=True)
        chkpt = os.path.join(saveDir, 'weights/auto/AE/AutoEncoder_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
    else:
        raise Exception
    cp_cb = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=5)
    # オーギュメントあり：
    # datagen.fit(x_train)
    # fit = model.fit_generator(datagen.flow(x_train, x_train, batch_size=batch_size), verbose=verbose, epochs=epochs, samples_per_epoch=x_train.shape[0],
    #                 validation_data=(x_val, x_val), callbacks=[cp_cb])  # tensorboard:tb_cb, checkpoint:cp_cb, EarlyStopping:es_cb
    # オーギュメントなし：
    fit = model.fit(x_train, x_train, batch_size=batch_size, verbose=verbose, epochs=epochs,
                    validation_split=validation_split, validation_data=(x_val, x_val), shuffle=True,
                    callbacks=[cp_cb])  # tensorboard:tb_cb, checkpoint:cp_cb, EarlyStopping:es_cb
    model_dir = "weights/auto/{}".format(mode_auto)
    model_name = get_latest_modified_file_path(model_dir)
    print(model_name, "を最良のモデルとします．")
    model = models.load_model(model_name)

    if show:
        loss = fit.history['loss']
        val_loss = fit.history['val_loss']

        # lossのグラフ
        plt.plot(range(len(loss)), loss, marker='.', label='train_loss')
        plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss')
        plt.legend(loc='best', fontsize=15)
        plt.grid()
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('loss', fontsize=15)
        # plt.ylim([0.20, 0.70])

        try:
            os.makedirs("figures/exp{}/loss/".format(experiment))
        except:
            pass
        try:
            os.makedirs("MODEL/exp{}/auto/".format(experiment))
        except:
            pass
        plt.savefig("figures/exp{}/loss/loss_{}_auto.eps".format(experiment, mode_auto))
        plt.savefig("figures/exp{}/loss/loss_{}_auto.png".format(experiment, mode_auto))
        model.save("MODEL/exp{}/auto/model_{}_auto.hdf5".format(experiment, mode_auto))
        # plot_model(model, "MODEL/auto/model_structure_{}_auto.png".format(mode_auto), show_shapes=True, show_layer_names=True)

        print("loss:eps, pngファイルを保存しました．")
        print("モデルを保存しました．")

    decoded_imgs = model.predict(x_train)

    try:
        os.makedirs("decoded_imgs/exp{}/{}".format(experiment, mode_auto))
    except:
        pass
    # 何個表示するか
    n = num_compare
    for i in range(n):
        # tmp = np.uint8(x_train[i] * 255).reshape((136, 196))  # reshape:gray用
        # tmp = Image.fromarray(tmp)
        # tmp.save('decoded_imgs/original_{}.png'.format(i))

        # オリジナルのテスト画像を表示
        Image.fromarray(np.uint8(x_train[i]*255).reshape(136, 196)).save('decoded_imgs/exp{}/{}/original_{}.png'.format(experiment, mode_auto, i))
        # 変換された画像を表示
        Image.fromarray(np.uint8(decoded_imgs[i]*255).reshape(136, 196)).save('decoded_imgs/exp{}/{}/decoded_{}.png'.format(experiment, mode_auto, i))


def train_test_split(data, label, n_class, train_rate=0.8):
    x_train = [data[i] for i in range(len(data)) if i % (len(data) / n_class) < (len(data) / n_class)*train_rate]
    x_test = [data[i] for i in range(len(data)) if i % (len(data) / n_class) >= (len(data) / n_class)*train_rate]
    y_train = [label[i] for i in range(len(label)) if i % (len(label) / n_class) < (len(data) / n_class)*train_rate]
    y_test = [label[i] for i in range(len(label)) if i % (len(label) / n_class) >= (len(data) / n_class)*train_rate]

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]

    test_img = [i for i in range(len(data)) if i % (len(data) / n_class) >= (len(data) / n_class)*train_rate]

    return (x_train, y_train), (x_test, y_test), test_img


def train_classify(mode_classify, mode_auto, data, label, n_class, input_shape, saveDir, batch_size=100, verbose=1, epochs=20,
              validation_split=0.2, show=False, experiment=None):
    (x_train, y_train), (x_test, y_test), test_img = train_test_split(data=data, label=label, n_class=n_class)

    def objective_mlp(trial):
        K.clear_session()
        mid_units = int(trial.suggest_discrete_uniform("mid_units", 100, 1000, 100))
        drop_rate = trial.suggest_uniform("drop_rate", 0, 0.5)

        model = model_mlp_classify(input_shape, n_class, mid_units, drop_rate)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs,
                        validation_split=validation_split, shuffle=True)
        test_score = model.evaluate(x_test, y_test, verbose=verbose)

        return test_score[1]

    def objective_cnn(trial):
        K.clear_session()
        num_layer = trial.suggest_int("num_layer", 2, 7)
        mid_units = int(trial.suggest_discrete_uniform("mid_units", 100, 500, 100))
        num_filters = [int(trial.suggest_discrete_uniform("num_filters" + str(i), 16, 128, 16)) for i in
                       range(num_layer)]
        activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])
        drop_rate = trial.suggest_uniform("drop_rate", 0, 0.5)

        model = model_simple_cnn_classify(input_shape, n_class, num_layer, activation, mid_units, num_filters,
                                          drop_rate)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs,
                        validation_split=validation_split, shuffle=True)
        test_score = model.evaluate(x_test, y_test, verbose=verbose)
        return test_score[1]

    study = optuna.create_study(direction="maximize")

    if mode_classify == "mlp_with_dist":
        study.optimize(objective_mlp, n_trials=20)
    elif mode_classify == "simple_cnn":
        study.optimize(objective_cnn, n_trials=20)
    print(study.best_params)
    if mode_classify == "mlp_with_dist":
        model = model_mlp_classify(input_shape, n_class, mid_units=int(study.best_params["mid_units"]), drop_rate=study.best_params["drop_rate"])
    elif mode_classify == "simple_cnn":
        num_filters = [int(study.best_params["num_filters"+str(i)]) for i in
                       range(study.best_params["num_layer"])]
        model = model_simple_cnn_classify(input_shape, n_class,
                                          num_layer=study.best_params["num_layer"],
                                          activation=study.best_params["activation"],
                                          mid_units=int(study.best_params["mid_units"]),
                                          num_filters=num_filters,
                                          drop_rate=study.best_params["drop_rate"])
    else:
        raise Exception

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=["accuracy"])

    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

    try:
        os.makedirs("tflog/exp{}/classify".format(experiment))
    except:
        pass

    if mode_auto == "CAE":
        tb_cb = TrainValTensorBoard(log_dir="tflog/exp{}/classify/{}".format(experiment, mode_auto), histogram_freq=1, write_graph=True)
        chkpt = os.path.join(saveDir, 'weights/classify/CAE/AutoEncoder_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
    elif mode_auto == "AE":
        tb_cb = TrainValTensorBoard(log_dir="tflog/exp{}/classify/{}".format(experiment, mode_auto), histogram_freq=1, write_graph=True)
        chkpt = os.path.join(saveDir, 'weights/classify/AE/AutoEncoder_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
    elif mode_classify == "simple_cnn":
        tb_cb = TrainValTensorBoard(log_dir="tflog/exp{}/classify/{}".format(experiment, mode_classify), histogram_freq=1, write_graph=True)
        chkpt = os.path.join(saveDir,
                             'weights/classify/simple_cnn/AutoEncoder_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
    else:
        raise Exception
    cp_cb = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=5)
    fit = model.fit(x_train, y_train, batch_size=batch_size, verbose=verbose, epochs=epochs,
                    validation_split=validation_split, shuffle=True,
                    callbacks=[cp_cb])  # tensorboard:tb_cb, checkpoint:cp_cb, EarlyStopping:es_cb

    if mode_auto == "AE" or mode_auto == "CAE":
        model_dir = "weights/classify/{}".format(mode_auto)
    elif mode_classify == "simple_cnn":
        model_dir = "weights/classify/{}".format(mode_classify)
    else:
        raise Exception
    model_name = get_latest_modified_file_path(model_dir)
    print(model_name, "を最良のモデルとします．")
    model = models.load_model(model_name)

    if show:
        loss = fit.history['loss']
        val_loss = fit.history['val_loss']

        # lossのグラフ
        plt.plot(range(len(loss)), loss, marker='.', label='train_loss')
        plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss')
        plt.legend(loc='best', fontsize=15)
        plt.grid()
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('loss', fontsize=15)

        if mode_classify == "simple_cnn":
            plt.savefig("figures/exp{}/loss/loss_{}_classify.eps".format(experiment, mode_classify))
            plt.savefig("figures/exp{}/loss/loss_{}_classify.png".format(experiment, mode_classify))
        else:
            plt.savefig("figures/exp{}/loss/loss_{}_classify.eps".format(experiment, mode_auto))
            plt.savefig("figures/exp{}/loss/loss_{}_classify.png".format(experiment, mode_auto))
        print("loss:eps, pngファイルを保存しました．")

        plt.figure()

        acc = fit.history['acc']
        val_acc = fit.history['val_acc']
        plt.plot(range(len(acc)), acc, marker='.', label='train_acc')
        plt.plot(range(len(val_acc)), val_acc, marker='.', label='val_acc')
        plt.legend(loc='best', fontsize=15)
        plt.grid()
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('acc', fontsize=15)
        plt.ylim([0, 1])
        try:
            os.makedirs("MODEL/exp{}/classify".format(experiment))
        except:
            pass

        try:
            os.makedirs("figures/exp{}/acc".format(experiment))
        except:
            pass
        if mode_classify == "simple_cnn":
            plt.savefig("figures/exp{}/acc/acc_{}_classify.eps".format(experiment, mode_classify))
            plt.savefig("figures/exp{}/acc/acc_{}_classify.png".format(experiment, mode_classify))
            model.save("MODEL/exp{}/classify/model_{}_classify.hdf5".format(experiment, mode_classify))
            # plot_model(model, "MODEL/classify/model_structure_simple_cnn.png", show_shapes=True, show_layer_names=True)
        else:
            plt.savefig("figures/exp{}/acc/acc_{}_classify.eps".format(experiment, mode_auto))
            plt.savefig("figures/exp{}/acc/acc_{}_classify.png".format(experiment, mode_auto))
            model.save("MODEL/exp{}/classify/model_{}.hdf5".format(experiment, mode_auto))
            # plot_model(model, "MODEL/classify/model_structure_MLP.png", show_shapes=True, show_layer_names=True)
        print("acc:eps, pngファイルを保存しました．")
        print("モデルを保存しました．")

    train_score = model.evaluate(x_train, y_train, verbose=verbose)
    test_score = model.evaluate(x_test, y_test, verbose=verbose)

    train_accuracy = train_score[1]
    test_accuracy = test_score[1]
    print("train accuracy : ", train_accuracy)
    print('test accuracy : ', test_accuracy)
    y_pred_test = model.predict_classes(x_test)
    return train_accuracy, test_accuracy, test_img, y_test, y_pred_test
