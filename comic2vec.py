from keras import models
from utils.loader import Loader
import os
import numpy as np
from keras.models import Model


class Comic2Vec:
    def __init__(self):
        pass

    def comic2vec(self, data, model_name, mode_out, mode_auto):
        """
        :param data:画像データ(numpy)
        :param model_name:使いたいmodelのhdfファイル．comic2vec.pyから相対パスを指定
        :param mode_out:raw:そのままのoutputを出力
                         hwf:フィルターも込みでFlattenにする
                         hw:フィルターを単純加算でFlattenにする
                mode_auto:オートエンコーダのモデル構造
                         CAE or AE
        :return: dataの分散表現(None, width, height, filters)のテンソル
        """
        model = models.load_model(model_name)
        layer_name = 'mid'
        if mode_auto == "CAE":
            data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
            # layers = model.layers[0:14]  # filter数変えたいときのやつ
        elif mode_auto == "AE":
            data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2],))
        else:
            raise Exception
        print(model.layers)
        activation_model = models.Model(input=model.input, outputs=model.get_layer(layer_name).output)
        out = activation_model.predict(data)
        print("分散表現のサイズは{}です．".format(out.shape))
        os.makedirs("distributed", exist_ok=True)

        if mode_auto == "AE":
            np.save("distributed/{}.npy".format(mode_auto), out)
            print("分散表現をnpy形式で保存しました．")
            return out

        elif mode_auto == "CAE":
            if mode_out == "raw":
                np.save("distributed/{}.npy".format(mode_auto), out)
                print("分散表現をnpy形式で保存しました．")
                return out
            elif mode_out == "hwf":
                out = out.reshape((out.shape[0], out.shape[1] * out.shape[2] * out.shape[3]))
                np.save("distributed/{}.npy".format(mode_auto), out)
                print("分散表現をnpy形式で保存しました．")
                return out
            elif mode_out == "hw":
                out = np.sum(out, axis=3)
                out = out.reshape((out.shape[0], out.shape[1] * out.shape[2]))
                np.save("distributed/{}.npy".format(mode_auto), out)
                print("分散表現をnpy形式で保存しました．")
                return out
            else:
                raise Exception
        else:
            raise Exception

        # 各層のoutputを取得したい場合
        # for i, activation in enumerate(activations):
        #     print("{}: {}".format(i, str(activation.shape)))


if __name__ == "__main__":
    mode_auto = "AE"
    # mode_auto = "CAE"

    mode_out = "hwf"
    load_dir = os.path.join(os.getcwd(), "imgs_param/ALL")

    model_name = "MODEL/auto/model_{}_auto.hdf5".format(mode_auto)

    loader_ins = Loader(load_dir)
    loader_ins.load(gray=True, size=(196, 136))  # 横×縦
    data = loader_ins.get_data(norm=True)

    C2V = Comic2Vec()
    output = C2V.comic2vec(data, model_name, mode_out=mode_out, mode_auto=mode_auto)
    print(output.shape)
