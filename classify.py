#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:52:48 2019

@author: barbaraxiong
"""
import numpy as np
import tensorflow as tf
import time
import build_image_data
from PIL import Image, ImageOps
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

# Prepare data
data_sets = build_image_data.load_data()

x_train = data_sets["images_train"]
y_train = data_sets["labels_train"]
x_test = data_sets["images_test"]
y_test = data_sets["labels_test"]


# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

nb_train_samples = 4000
nb_validation_samples = 1000
epochs = 50
batch_size = 32

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# Fit the model
history = model.fit(x_train, y_train, epochs=500, batch_size=5, verbose=2, callbacks=callbacks_list)
# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
