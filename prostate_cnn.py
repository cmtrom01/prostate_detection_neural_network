#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:40:19 2019

@author: christrombley
"""

import tensorflow  as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
import keras
import keras.utils
from keras import utils as np_utils
from keras.utils import to_categorical
from google.colab import files
from google.colab import auth
from google.colab import drive
from oauth2client.client import GoogleCredentials
import tensorflow  as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
import keras
import keras.utils
from keras import utils as np_utils
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import tensorflow as tf
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mimg
%matplotlib inline
plt.rcParams["figure.figsize"] = (10,7)
from PIL import Image
from scipy import misc
import os
from keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle



##import google drive files
drive.mount('/content/gdrive/')
much_data = np.load('/content/gdrive/My Drive/imgFileNew.npy')
labels = np.load('/content/gdrive/My Drive/labelsNew.npy')



##declare variables
BATCH_NORM = True
batch_size = 64
num_classes = 2
epochs = 20
data_augmentation = True
BS = 32
EPOCHS = 20
classes = 2
chanDim = -1



##function that builds VGG model and returns it  
class SmallerVGGNet:
    @staticmethod
    def build():
        
          model = Sequential()

          model.add(Conv2D(64, (3, 3), padding='same', input_shape=(50,50,1), name='block1_conv1'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

          model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))
          model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

          model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv4'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

          model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv4'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))
          model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

          model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv4'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))

          model.add(Flatten())

          model.add(Dense(4096))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))
          model.add(Dropout(0.5))

          model.add(Dense(4096, name='fc2'))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('relu'))
          model.add(Dropout(0.5))

          model.add(Dense(num_classes))
          model.add(BatchNormalization()) if BATCH_NORM else None
          model.add(Activation('softmax'))

          sgd = keras.optimizers.SGD(lr=0.05, decay=0, nesterov=True)

          model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy", auc, specificity, recall, precision])
          return model

      
      

# scale the raw pixel intensities to the range [0, 1] bc pixels are 0 to 255
data = np.array(much_data, dtype="float") ##/ 4630.0
labels = np.array(labels)



(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.3, random_state = 48)

##first num should be number of images
trainX = trainX.reshape([-1,50, 50,1])
testX = testX.reshape([-1,50, 50,1])

 
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)
 

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")



# initialize the model
print("[INFO] compiling model...")
model = SmallerVGGNet.build()
 
# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("/content/gdrive/My Drive/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/content/gdrive/My Drive/model.h5")
print("Saved model to disk")

