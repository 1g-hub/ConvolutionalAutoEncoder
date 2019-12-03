# -*- coding: shift-jis -*-
from loader import Loader
from model import *
import os
import tensorflow as tf
from tensorflow.python import keras
from comic2vec import Comic2Vec


def encode_and_decode(mode_auto, weights_dir, load_dir):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    old_session = KTF.get_session()

    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    loader_ins = Loader(load_dir)
    loader_ins.load(gray=True, size=(196, 136))  # 横×縦
    data = loader_ins.get_data(norm=True)  # (None, Height, Width)

    if mode_auto == "CAE":
        input_shape = (data.shape[1], data.shape[2], 1)
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))  # (None, Height, Width, 1)
    elif mode_auto == "AE":
        input_shape = (data.shape[1]*data.shape[2],)
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2],))  # (None, Height*Width, 1)
    else:
        raise Exception

    x_train = data[:len(data) * 0.8]
    x_val = data[len(data) * 0.8:]

    train_auto(mode_auto=mode_auto, x_train=x_train, x_val=x_val, input_shape=input_shape, weights_dir=weights_dir, batch_size=32, verbose=1, epochs=500,
                  validation_split=0.1, show=True, num_compare=10)

    data = loader_ins.get_data(norm=True)
    C2V = Comic2Vec()
    model_name = get_latest_modified_file_path(weights_dir)
    print(model_name, "をモデルとして分散表現化します．")
    C2V.comic2vec(data, model_name, mode_auto=mode_auto, mode_out="hwf")

    KTF.set_session(old_session)


if __name__ == "__main__":
    # mode_auto = "CAE"
    mode_auto = "AE"

    load_dir = os.path.join(os.getcwd(), "images/")

    weights_dir = "weights/auto/{}".format(mode_auto)

    encode_and_decode(mode_auto, weights_dir=weights_dir, load_dir=load_dir)
