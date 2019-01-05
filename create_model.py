#create net model CNN
#Gubin M.

from __future__ import print_function

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.optimizers import RMSprop
import h5py
import os

#path_save_model
model_path='new4_model_3levels-4.h5'

#keras parametrs
num_classes = 2
img_rows, img_cols = 513, 219
input_shape = (img_rows, img_cols, 1)

#create Sequential model
model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(324, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Save model and weights
model.save(model_path)
##model.save_weights(model_weights_path)
print('Saved no-trained model at %s ' % model_path)

