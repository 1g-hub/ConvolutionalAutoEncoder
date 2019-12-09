import numpy as np
import os
from keras import models
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.layers import UpSampling2D
from keras.callbacks import ModelCheckpoint
import matplotlib
from glob import glob
matplotlib.use("PS")
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from PIL import Image


def get_latest_modified_file_path(model_dir):
  target = os.path.join(model_dir, '*')
  files = [(f, os.path.getmtime(f)) for f in glob(target)]
  latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
  return latest_modified_file_path[0]


class TrainValTensorBoard(TensorBoard):
    """
    Tensorboard においてtrainとvalidationのロスを同時にプロットする関数
    """
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


def train_auto(mode_auto, x_train, x_val, input_shape, weights_dir, batch_size=8, verbose=1, epochs=20, num_compare=2):
    if mode_auto == "CAE":
        model = model_cnn_auto(input_shape)
    elif mode_auto == "AE":
        model = model_mlp_auto(input_shape)
    else:
        raise Exception

    model.compile(optimizer='Adam', loss='binary_crossentropy')

    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

    os.makedirs("tflog/auto", exist_ok=True)
    os.makedirs("weights/auto/{}".format(mode_auto), exist_ok=True)
    if mode_auto == "CAE":
        tb_cb = TrainValTensorBoard(log_dir="tflog/auto/{}".format(mode_auto), histogram_freq=1, write_graph=True)
        chkpt = 'weights/auto/CAE/AutoEncoder_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    elif mode_auto == "AE":
        tb_cb = TrainValTensorBoard(log_dir="tflog/auto/{}".format(mode_auto), histogram_freq=1, write_graph=True)
        chkpt = 'weights/auto/AE/AutoEncoder_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    else:
        raise Exception

    cp_cb = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=5)
    fit = model.fit(x_train, x_train, batch_size=batch_size, verbose=verbose, epochs=epochs, validation_data=(x_val, x_val), shuffle=True,
                    callbacks=[cp_cb])  # tensorboard:tb_cb, checkpoint:cp_cb, EarlyStopping:es_cb
    weights_name = get_latest_modified_file_path(weights_dir)
    print(weights_name, "を最良のモデル重みとします．")
    model = models.load_model(weights_name)

    loss = fit.history['loss']
    val_loss = fit.history['val_loss']

    # lossのグラフ
    plt.plot(range(len(loss)), loss, marker='.', label='train_loss')
    plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best', fontsize=15)
    plt.grid()
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('loss', fontsize=15)

    os.makedirs("figures/loss/", exist_ok=True)
    os.makedirs("MODEL/auto/", exist_ok=True)

    plt.savefig("figures/loss/loss_{}_auto.eps".format(mode_auto))
    plt.savefig("figures/loss/loss_{}_auto.png".format(mode_auto))
    model.save("MODEL/auto/model_{}_auto.hdf5".format(mode_auto))
    # モデル表示用(サーバでは使わない)
    # plot_model(model, "MODEL/auto/model_structure_{}_auto.png".format(mode_auto), show_shapes=True, show_layer_names=True)

    print("loss:eps, pngファイルを保存しました．")
    print("モデルを保存しました．")

    decoded_imgs = model.predict(x_train)

    os.makedirs("decoded_imgs/{}".format(mode_auto), exist_ok=True)
    # 何個表示するか
    for i in range(num_compare):
        # オリジナルのテスト画像を表示
        Image.fromarray(np.uint8(x_train[i]*255).reshape(136, 196)).save('decoded_imgs/{}/{}_original.png'.format(mode_auto, i))
        # 変換された画像を表示
        Image.fromarray(np.uint8(decoded_imgs[i]*255).reshape(136, 196)).save('decoded_imgs/{}/{}_decoded.png'.format(mode_auto, i))
