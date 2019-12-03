# -*- coding: utf-8 -*-
from comic2vec import Comic2Vec
import numpy as np
from loader import Loader
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import seaborn as sns


def t_sne(mode, mode_auto, mode_out, img_load_dir):
    if mode == "create":
        model_name = "MODEL/auto/model_{}_auto.hdf5".format(mode_auto)
        loader_ins = Loader(img_load_dir)
        loader_ins.load(gray=True, size=(196, 136))  # 横×縦
        data = loader_ins.get_data(norm=True)
        C2V = Comic2Vec()
        output = C2V.comic2vec(data, model_name, mode_out=mode_out, mode_auto=mode_auto)
        print(output.shape)

    elif mode == "use_created" and mode_auto == "AE":
        output = np.load("distributed/{}.npy".format(mode_auto))
        print(output.shape)

    elif mode == "use_created" and mode_auto == "CAE":
        output = None
        output = np.load("distributed/{}.npy".format(mode_auto))
        print(output.shape)

    else:
        raise Exception

    perplexity = 80

    x_reduced = TSNE(n_components=2, perplexity=perplexity, random_state=0, learning_rate=100.0, n_iter=2000).fit_transform(output)
    x_reduced = np.array(x_reduced)

    id = [i for i in range(output.shape[0])]

    d1 = x_reduced[0:80]
    d1_id = id[0:80]
    d2 = x_reduced[80:160]
    d2_id = id[80:160]
    d3 = x_reduced[160:240]
    d3_id = id[160:240]

    for (i, j, k) in zip(d1[:, 0], d1[:, 1], d1_id):
        if k == 0:
            plt.plot(i, j, 'o', color="#984ea3", ms=4, mew=0.5, label="0")
            plt.annotate(k, xy=(i, j), fontsize=6)
        else:
            plt.plot(i, j, 'o', color="#984ea3", ms=4, mew=0.5)
            plt.annotate(k, xy=(i, j), fontsize=6)
    for (i, j, k) in zip(d2[:, 0], d2[:, 1], d2_id):
        if k == 80:
            plt.plot(i, j, 'o', color="#00cccc", ms=4, mew=0.5, label="1")
            plt.annotate(k, xy=(i, j), fontsize=6)
        else:
            plt.plot(i, j, 'o', color="#00cccc", ms=4, mew=0.5)
            plt.annotate(k, xy=(i, j), fontsize=6)
    for (i, j, k) in zip(d3[:, 0], d3[:, 1], d3_id):
        if k == 160:
            plt.plot(i, j, 'o', color="#ff0000", ms=4, mew=0.5, label="2")
            plt.annotate(k, xy=(i, j), fontsize=6)
        else:
            plt.plot(i, j, 'o', color="#ff0000", ms=4, mew=0.5)
            plt.annotate(k, xy=(i, j), fontsize=6)

    plt.legend(fontsize=17)
    if mode_auto == "AE":
        plt.savefig(save_dir + "t_sne_{}_{}.eps".format(mode_auto, perplexity))
        plt.savefig(save_dir + "t_sne_{}_{}.png".format(mode_auto, perplexity))
    else:
        plt.savefig(save_dir + "t_sne_{}_{}_{}.eps".format(mode_auto, mode_out, perplexity))
        plt.savefig(save_dir + "t_sne_{}_{}_{}.png".format(mode_auto, mode_out, perplexity))
    print("t_sne図を保存しました．")


if __name__ == "__main__":
    # mode: create: モデル重みから分散表現を作る
    #       use_created: 分散表現を使う
    # mode = "create"
    mode = "use_created"

    mode_auto = "CAE"
    # mode_auto = "AE"

    mode_out = "hwf"
    # mode_out = "hw"
    # mode_out = "raw"

    save_dir = "figures/t_sne"
    os.makedirs(save_dir, exist_ok=True)

    img_load_dir = os.path.join(os.getcwd(), "images")

    t_sne(mode, mode_auto=mode_auto, mode_out=mode_out, img_load_dir=img_load_dir)
