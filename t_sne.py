# -*- coding: utf-8 -*-
from comic2vec import Comic2Vec
import numpy as np
from utils.loader import Loader
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def t_sne(mode, mode_auto, mode_out, load_dir, n_components, experiment, weight_all=False):
    if mode == "create":
        model_name = "MODEL/exp{}/auto/model_{}_auto.hdf5".format(experiment, mode_auto)
        loader_ins = Loader(load_dir)
        loader_ins.load(gray=True, size=(196, 136))  # 横×縦
        data = loader_ins.get_data(norm=True)
        C2V = Comic2Vec()
        output = C2V.comic2vec(data, model_name, mode_out=mode_out, mode_auto=mode_auto, experiment=experiment)
        print(output.shape)

    elif mode == "use_created" and mode_auto == "AE":
        output = None
        for exp in experiment:
            if output is None:
                if weight_all:
                    output = np.load("distributed/exp{}/{}_weight_all.npy".format(exp, mode_auto))
                else:
                    output = np.load("distributed/exp{}/{}.npy".format(exp, mode_auto))
            else:
                if weight_all:
                    tmp = np.load("distributed/exp{}/{}_weight_all.npy".format(exp, mode_auto))
                else:
                    tmp = np.load("distributed/exp{}/{}.npy".format(exp, mode_auto))
                output = np.concatenate([output, tmp], axis=0)
        print(output.shape)
    elif mode == "use_created" and mode_auto == "CAE":
        output = None
        for exp in experiment:
            if output is None:
                if weight_all:
                    output = np.load("distributed/exp{}/{}_weight_all.npy".format(exp, mode_auto))
                else:
                    output = np.load("distributed/exp{}/{}.npy".format(exp, mode_auto))
            else:
                if weight_all:
                    tmp = np.load("distributed/exp{}/{}_weight_all.npy".format(exp, mode_auto))
                else:
                    tmp = np.load("distributed/exp{}/{}.npy".format(exp, mode_auto))
                output = np.concatenate([output, tmp], axis=0)
        print(output.shape)
    else:
        raise Exception

    perplexity = 80

    x_reduced = TSNE(n_components=n_components, perplexity=perplexity, random_state=0, learning_rate=100.0, n_iter=2000).fit_transform(output)
    x_reduced = np.array(x_reduced)

    if n_components == 2:
        id = [i for i in range(output.shape[0])]

        if len(experiment) == 1:
            if experiment[0] == "1_4":
                d1 = x_reduced[0:64]
                d1_id = id[0:64]
                d2 = x_reduced[64:138]
                d2_id = id[64:138]
                d3 = x_reduced[138:204]
                d3_id = id[138:204]
            elif experiment[0] == "1_5":
                d1 = x_reduced[0:63]
                d1_id = id[0:63]
                d2 = x_reduced[62:126]
                d2_id = id[62:126]
                d3 = x_reduced[126:195]
                d3_id = id[126:195]
            else:
                d1 = x_reduced[0:80]
                d1_id = id[0:80]
                d2 = x_reduced[80:160]
                d2_id = id[80:160]
                d3 = x_reduced[160:240]
                d3_id = id[160:240]

            for (i, j, k) in zip(d1[:, 0], d1[:, 1], d1_id):
                if k == 0:
                    plt.plot(i, j, 'o', color="#984ea3", ms=4, mew=0.5, label="moe")
                    plt.annotate(k, xy=(i, j), fontsize=6)
                else:
                    plt.plot(i, j, 'o', color="#984ea3", ms=4, mew=0.5)
                    plt.annotate(k, xy=(i, j), fontsize=6)
            for (i, j, k) in zip(d2[:, 0], d2[:, 1], d2_id):
                if k == 80:
                    plt.plot(i, j, 'o', color="#00cccc", ms=4, mew=0.5, label="seinen")
                    plt.annotate(k, xy=(i, j), fontsize=6)
                else:
                    plt.plot(i, j, 'o', color="#00cccc", ms=4, mew=0.5)
                    plt.annotate(k, xy=(i, j), fontsize=6)
            for (i, j, k) in zip(d3[:, 0], d3[:, 1], d3_id):
                if k == 160:
                    plt.plot(i, j, 'o', color="#ff0000", ms=4, mew=0.5, label="shonen")
                    plt.annotate(k, xy=(i, j), fontsize=6)
                else:
                    plt.plot(i, j, 'o', color="#ff0000", ms=4, mew=0.5)
                    plt.annotate(k, xy=(i, j), fontsize=6)
        else:  # パーツ抜きの変化の遷移を見る場合
            d1 = x_reduced[0:240]
            d1_id = list(range(0, 240))
            d2 = x_reduced[240:480]
            d2_id = list(range(240, 480))

            for (i, j, k) in zip(np.array(d1)[:, 0], np.array(d1)[:, 1], d1_id):
                if k == 0:
                    plt.plot(i, j, 'o', color="#cc00cc", ms=2, mew=0.5, label="ALL")
                    plt.annotate(k, xy=(i, j), fontsize=6)
                elif k == 1:
                    plt.plot(i, j, 'o', color="red", ms=2, mew=0.5)
                    plt.annotate(k, xy=(i, j), fontsize=6)
                else:
                    plt.plot(i, j, 'o', color="#cc00cc", ms=2, mew=0.5)
                    plt.annotate(k, xy=(i, j), fontsize=6)
            for (i, j, k) in zip(np.array(d2)[:, 0], np.array(d2)[:, 1], d2_id):
                if k == 240:
                    plt.plot(i, j, 'o', color="#00cccc", ms=2, mew=0.5, label="REM_EYE")
                    plt.annotate(k, xy=(i-2, j), fontsize=6)
                elif k == 241:
                    plt.plot(i, j, 'o', color="red", ms=2, mew=0.5)
                    plt.annotate(k, xy=(i-2, j), fontsize=6)
                else:
                    plt.plot(i, j, 'o', color="#00cccc", ms=2, mew=0.5)
                    plt.annotate(k, xy=(i-2, j), fontsize=6)
        plt.legend(fontsize=17)
        if mode_auto == "AE":
            if weight_all:
                plt.savefig(save_dir + "t_sne_{}_{}_weight_all.eps".format(mode_auto, perplexity))
                plt.savefig(save_dir + "t_sne_{}_{}_weight_all.png".format(mode_auto, perplexity))
            else:
                plt.savefig(save_dir + "t_sne_{}_{}.eps".format(mode_auto, perplexity))
                plt.savefig(save_dir + "t_sne_{}_{}.png".format(mode_auto, perplexity))
        else:
            if weight_all:
                plt.savefig(save_dir + "t_sne_{}_{}_{}_weight_all.eps".format(mode_auto, mode_out, perplexity))
                plt.savefig(save_dir + "t_sne_{}_{}_{}_weight_all.png".format(mode_auto, mode_out, perplexity))
            else:
                plt.savefig(save_dir + "t_sne_{}_{}_{}.eps".format(mode_auto, mode_out, perplexity))
                plt.savefig(save_dir + "t_sne_{}_{}_{}.png".format(mode_auto, mode_out, perplexity))
        print("t_sne図を保存しました．")
    elif n_components == 3:
        d1 = x_reduced[0:80]
        d2 = x_reduced[80:160]
        d3 = x_reduced[160:240]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.plot(d1[:, 0], d1[:, 1], d1[:, 2], "o", color="#984ea3", ms=4, mew=0.5, label="moe")
        ax.plot(d2[:, 0], d2[:, 1], d2[:, 2], "o", color="#00cccc", ms=4, mew=0.5, label="seinen")
        ax.plot(d3[:, 0], d3[:, 1], d3[:, 2], "o", color="#ff0000", ms=4, mew=0.5, label="shonen")
        plt.legend()
        # ax.scatter(data=df, x="x", y="y", z="z", hue="touch")
        plt.show()
    else:
        raise Exception


if __name__ == "__main__":
    # mode = "create"
    mode = "use_created"

    mode_auto = "CAE"
    # mode_auto = "AE"

    mode_out = "hwf"
    # mode_out = "hw"
    # mode_out = "raw"

    """
    experiment=1_1:設定ALL
    experiment=1_2:設定REMOVE_EYE
    experiment=1_3:設定REMOVE_MOUTH
    experiment=1_4:設定EYE_ONLY
    experiment=1_5:設定MOUTH_ONLY
    """
    experiment = ["1_4"]
    try:
        os.makedirs("figures/exp{}/t_sne".format(experiment[0]))
    except:
        pass

    if len(experiment) == 1 and experiment[0] == "1_1":
        load_dir = os.path.join(os.getcwd(), "imgs_param/ALL")
        save_dir = "figures/exp{}/t_sne/".format(experiment[0])
    elif len(experiment) == 1 and experiment[0] == "1_2":
        load_dir = os.path.join(os.getcwd(), "imgs_param/REMOVE_EYE")
        save_dir = "figures/exp{}/t_sne/".format(experiment[0])
    elif len(experiment) == 1 and experiment[0] == "1_3":
        load_dir = os.path.join(os.getcwd(), "imgs_param/REMOVE_MOUTH")
        save_dir = "figures/exp{}/t_sne/".format(experiment[0])
    elif len(experiment) == 1 and experiment[0] == "1_4":
        load_dir = os.path.join(os.getcwd(), "imgs_param/EYE_ONLY")
        save_dir = "figures/exp{}/t_sne/".format(experiment[0])
    elif len(experiment) == 1 and experiment[0] == "1_5":
        load_dir = os.path.join(os.getcwd(), "imgs_param/MOUTH_ONLY")
        save_dir = "figures/exp{}/t_sne/".format(experiment[0])
    else:
        load_dir = None
        save_dir = "figures/exp_ALL_and_REMOVE_EYE/"
    weight_all = False
    # n_components = 3
    n_components = 2

    t_sne(mode, mode_auto=mode_auto, mode_out=mode_out, load_dir=load_dir, n_components=n_components, experiment=experiment, weight_all=weight_all)
