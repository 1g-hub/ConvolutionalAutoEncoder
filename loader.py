# -*- coding: utf-8 -*-
#  画像ファイルを読み込み，ラベルとデータに格納し，
# 学習用データセットと検証用データセットをラベルとともに返す関数を作成する
import numpy as np
import os
from PIL import Image


class Loader:
    def __init__(self, path):
        self.dir = path
        self.imgs = []

    def get_data(self, norm=True):
        if len(self.imgs) == 0:
            print("Error:loader:get_data")
        elif norm:
            return np.array(self.imgs) / 255.
        else:
            return np.array(self.imgs)

    def load(self, gray=False, size=(18, 18)):
        """
        :param gray:グレースケールにするかどうか
        :param size:(横, 縦)の長さのタプル
        """
        for file in sorted(os.listdir(self.dir)):
            file_path = os.path.join(self.dir, file)
            if gray:
                tmp = Image.open(file_path).convert('LA')
                mask = tmp.split()[-1]
                canvas = Image.new('L', size, (255,))
                canvas.paste(tmp, None, mask=mask)
                self.imgs.append(np.array(canvas))
            else:
                img = Image.open(file_path)
                self.imgs.append(np.array(img))
