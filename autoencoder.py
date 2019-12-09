# -*- coding: shift-jis -*-
from loader import Loader
from model import *
import os
import tensorflow as tf
from tensorflow.python import keras
from image2vec import img2vec
import keras.backend.tensorflow_backend as KTF


def encode_and_decode(mode_auto, exp_condition):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    keras.backend.set_session(sess)

    old_session = KTF.get_session()

    session = tf.compat.v1.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)

    loader_ins = Loader(exp_condition["load_dir"])
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

    x_train = data[:int(len(data) * exp_condition["train_rate"])]
    x_val = data[int(len(data) * exp_condition["train_rate"]):]

    train_auto(mode_auto=mode_auto,
               x_train=x_train,
               x_val=x_val,
               input_shape=input_shape,
               weights_dir=exp_condition["weights_dir"],
               batch_size=exp_condition["batch_size"],
               verbose=1,
               epochs=exp_condition["epochs"],
               num_compare=2
               )

    data = loader_ins.get_data(norm=True)
    model_name = get_latest_modified_file_path(exp_condition["weights_dir"])
    print(model_name, "をモデルとして分散表現化します．")
    img2vec(data, model_name, mode_auto=mode_auto, mode_out="hwf")

    KTF.set_session(old_session)


if __name__ == "__main__":
    # パラメータをハードコーディングしているのでよしなにjson形式にするなりargparseにするなり．
    mode_auto = "CAE"

    exp_condition = {
        "load_dir": os.path.join(os.getcwd(), "images/"),
        "weights_dir": "weights/auto/{}".format(mode_auto),
        "epochs": 50,
        "batch_size": 16,
        "train_rate": 0.8
    }
    encode_and_decode(mode_auto, exp_condition)
