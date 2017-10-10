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

img_width = 320
img_heigth = 320


#for filename in glob.glob("/Users/ronny/Downloads/__new/Stanford Dogs Dataset/example/*"):
#	if os.path.isfile(filename):
#		img = imread(filename)
#		img = imresize(img, (img_width, img_heigth))
#		imsave(filename, img)

filelist = glob.glob('/Users/ronny/Downloads/__new/Stanford Dogs Dataset/example/*')
dogs = np.array([np.array(imread(filename)) for filename in filelist])
dogs_y = np.zeros((1000,1))
filelist = glob.glob('/Users/ronny/Downloads/__new/Cats Dataset/example/*')
cats = np.array([np.array(imread(filename)) for filename in filelist])
cats_y = np.ones((1000,1))

both = np.concatenate((dogs, cats), axis=0)
both_y = np.concatenate((dogs_y, cats_y), axis=0)


np.random.seed(1)
np.random.shuffle(both)
np.random.seed(1)
np.random.shuffle(both_y)
imsave("out1.jpg", both[0])
imsave("out2.jpg", both[1])
imsave("out3.jpg", both[2])
imsave("out4.jpg", both[3])
imsave("out5.jpg", both[4])
imsave("out6.jpg", both[5])

both_y[0:6]
# funktioniert!!!

print(str(dogs.shape))
print(str(cats.shape))

sys.exit(0)



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

#model.fit_generator(training_data, steps_per_epoch=500, epochs=10, validation_data=validation_data)

#model.save_weights ...

print("Yeah")
