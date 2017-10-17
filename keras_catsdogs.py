#!/usr/bin/env python3
#coding=UTF-8

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from scipy.misc import imread, imsave, imresize
import glob
import numpy as np

batch_size = 100
num_classes = 2
epochs = 10

# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3

# load my custom data
folder = "catsdogsdataset30k"
dogs_folder = "dogs"
cats_folder = "cats"


filelist = glob.glob(folder + '/' + dogs_folder + '/*')
train_dogs = np.array([np.array(imread(filename).flatten()) for filename in filelist])
train_dogs_y = np.zeros((len(filelist),1))
filelist = glob.glob(folder + '/' + dogs_folder + '_test/*')
test_dogs = np.array([np.array(imread(filename).flatten()) for filename in filelist])
test_dogs_y = np.zeros((len(filelist),1))

filelist = glob.glob(folder + '/' + cats_folder + '/*')
train_cats = np.array([np.array(imread(filename).flatten()) for filename in filelist])
train_cats_y = np.ones((len(filelist),1))
filelist = glob.glob(folder + '/' + cats_folder + '_test/*')
test_cats = np.array([np.array(imread(filename).flatten()) for filename in filelist])
test_cats_y = np.ones((len(filelist),1))

train_both = np.concatenate((train_dogs, train_cats), axis=0)
train_both_y = np.concatenate((train_dogs_y, train_cats_y), axis=0)
test_both = np.concatenate((test_dogs, test_cats), axis=0)
test_both_y = np.concatenate((test_dogs_y, test_cats_y), axis=0)

np.random.seed(1)
np.random.shuffle(train_both)
np.random.seed(1)
np.random.shuffle(train_both_y)
np.random.seed(2)
np.random.shuffle(test_both)
np.random.seed(2)
np.random.shuffle(test_both_y)


# the data, shuffled and split between train and test sets
x_train = train_both.reshape(train_both.shape[0], img_rows,img_cols,img_channels)
y_train = train_both_y
x_test = test_both.reshape(test_both.shape[0], img_rows,img_cols,img_channels)
y_test = test_both_y


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices (-> one-hot-vectors)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# wiederhole [convnets](https://www.youtube.com/watch?v=GYGYnspV230)
# siehe <https://keras.io/layers/convolutional/#conv2d>

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
# hier kann kein Conv2D mehr eingebaut werden
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
