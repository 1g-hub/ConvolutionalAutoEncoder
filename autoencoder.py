# -*- coding: shift-jis -*-
from utils.loader import Loader
from utils.model import *
import os
import tensorflow as tf
from tensorflow.python import keras
from comic2vec import Comic2Vec


def encode_and_decode(mode_auto, n_class, model_dir, load_dir, experiment):
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
    # loader_ins.load(gray=True, size=(208, 128))  # 横×縦
    data = loader_ins.get_data(norm=True)  # (None, 136, 196)

    save_dir = os.getcwd()

    if mode_auto == "CAE":
        input_shape = (data.shape[1], data.shape[2], 1)
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))  # (None, 136, 196, 1)
    elif mode_auto == "AE":
        input_shape = (data.shape[1]*data.shape[2],)
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2],))  # (None, 136*196, 1)
    else:
        raise Exception

    train_auto(mode_auto=mode_auto, data=data, n_class=n_class, input_shape=input_shape, batch_size=32, verbose=1, epochs=500,
                  validation_split=0.1, show=True, saveDir=save_dir, num_compare=10, experiment=experiment)

    data = loader_ins.get_data(norm=True)
    C2V = Comic2Vec()
    model_name = get_latest_modified_file_path(model_dir)
    print(model_name, "をモデルとして分散表現化します．")
    C2V.comic2vec(data, model_name, mode_auto=mode_auto, mode_out="hwf", experiment=experiment)

    KTF.set_session(old_session)


if __name__ == "__main__":
    # mode_auto = "CAE"
    mode_auto = "AE"

    """
    experiment=1_1:設定ALL
    experiment=1_2:設定REMOVE_EYE
    experiment=1_3:設定REMOVE_MOUTH
    experiment=1_4:設定EYE_ONLY
    experiment=1_5:設定MOUTH_ONLY
    experiment=3:filter数を変える
    """
    experiment = "1_4"

    if experiment == "1_1" or experiment == "3":
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

    n_class = 3

    model_dir = "weights/auto/{}".format(mode_auto)

    encode_and_decode(mode_auto, n_class, model_dir=model_dir, load_dir=load_dir, experiment=experiment)
