# -*- coding: utf-8 -*-
#  画像ファイルを読み込み，ラベルとデータに格納し，
# 学習用データセットと検証用データセットをラベルとともに返す関数を作成する
import numpy as np
import cv2 as cv
import os
from PIL import Image, ImageOps
from keras.utils import np_utils


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
        for p_folder in sorted(os.listdir(self.dir)):
            parent_path = os.path.join(self.dir, p_folder)
            for folder_ in os.listdir(parent_path):
                child_path = os.path.join(parent_path, folder_)
                for i, file in enumerate(os.listdir(child_path)):
                    file_path = os.path.join(child_path, file)
                    if gray:
                        tmp = Image.open(file_path).convert('LA')
                        mask = tmp.split()[-1]
                        canvas = Image.new('L', size, (255,))
                        canvas.paste(tmp, None, mask=mask)
                        self.imgs.append(np.array(canvas))
                    else:
                        img = Image.open(file_path)
                        self.imgs.append(np.array(img))
