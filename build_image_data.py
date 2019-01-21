#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:24:35 2019

@author: barbaraxiong
"""

import imageio
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from random import shuffle

def load_batch(foldername):
    '''load data from single folder'''
    images = []
    labels = []
    for category in os.listdir(foldername):
        if os.path.isdir(os.path.join(foldername, category)):
            for img in os.listdir(os.path.join(foldername, category)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    image = Image.open(os.path.join(foldername, category)+"/"+img)
                    sess = tf.Session()
                    with sess.as_default():
                        images.append(np.asarray(image))
                    labels.append(str(category))
    print(np.asarray(images))
    return np.asarray(images), np.asarray(labels)

def load_data():
    '''load all folder data and merge training batches'''
    X, Y = load_batch("colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/")
    xs = X
    ys = Y
    xs = np.concatenate([xs])
    ys = np.concatenate([ys])
    classes = ['01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']

    # Normalize Data

    #mean_image = np.mean(x_train, axis=0)
    #x_train = np.subtract(x_train, mean_image, out=x_train, casting = "unsafe")
    #x_test = np.subtract(x_test, mean_image, out=x_test, casting = "unsafe")

    data_dict = {
        'images': xs,
        'labels': ys
        }
    return data_dict

def main():
    data_sets = load_data()

    
if __name__ == '__main__':
  main()

