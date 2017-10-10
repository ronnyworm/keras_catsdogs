#!/usr/bin/env python3
#coding=UTF-8

# modified from https://www.youtube.com/watch?v=cAICT4Al5Ow

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Dropout, Flatten
from keras import backend as K
from scipy.misc import imread, imsave, imresize
import numpy as np
import glob
import sys
import os.path

img_width = 32
img_heigth = 32


# change image size
for filename in glob.glob("/Users/ronny/Downloads/__new/Cats Dataset/example32/*"):
	if os.path.isfile(filename):
		img = imread(filename)
		img = imresize(img, (img_width, img_heigth))
		imsave(filename, img)

print("done")

sys.exit()

filelist = glob.glob('/Users/ronny/Downloads/__new/Stanford Dogs Dataset/example320/*')

dogs = np.array([np.array(imread(filename)) for filename in filelist])
dogs_y = np.zeros((len(filelist),1))
filelist = glob.glob('/Users/ronny/Downloads/__new/Cats Dataset/example320/*')
cats = np.array([np.array(imread(filename)) for filename in filelist])
cats_y = np.ones((len(filelist),1))

both = np.concatenate((dogs, cats), axis=0)
both_y = np.concatenate((dogs_y, cats_y), axis=0)

np.random.seed(1)
np.random.shuffle(both)
np.random.seed(1)
np.random.shuffle(both_y)

print("done loading data")

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(img_width, img_heigth, 3)))
model.add(Activation(K.relu))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation(K.relu))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation(K.relu))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation(K.relu))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation(K.sigmoid))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print("done compiling the model")

model.fit(both, both_y, epochs=10, verbose=1)

#model.save_weights ...

print("done with fitting")
