# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 01:09:19 2020

@author: Abdul
"""



##### problems #####
# problem of images size in conv2d layer and training model function 

#import os
import numpy as np
#import cv2

##### preprocess ##### 
#from keras.preprocessing.image import ImageDataGenerator

#####  CNN model libraries ##### 
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam ## SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint #,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D
# import tensorflow as tf
#import tensorflow_addons as tfa

global image_size
def get_image_size(img):
    # since our image will be 28X28 pixel so we set the width and height same
    #image_size, image_size = img.shape()
    #return image_size
    pass



def Building_cnn_model(image_size):
    model  = Sequential()
    
    # building only 3 conv2D layers instead of original 4 conv2d layers model
    model.add(Conv2D(filter=32, kernel_size=(5,5), padding = 'Same', activation='relu', input_shape= (image_size, image_size, 1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filter=64, kernel_size=(3,3), padding = 'Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filter=96, kernel_size=(3,3), padding = 'Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filter=96, kernel_size=(2,2), padding = 'Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = 'softmax'))
    
    model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    
    file_path = "model_wieghts.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', weights_only=False, period=1)
    callbacks = [checkpoint]
    
    return model, callbacks

from prepDataset import get_dataset, get_classes
def model_training():
    img_dataset = get_dataset()
    labels = get_classes()
    model, callbacks = Building_cnn_model()
    sumaary = model.summary()
    
    img_size, img_size = img_dataset[0].shape
    # reshape images if needed
    y_data = to_categorical(labels)
    x_data = np.reshape((img_dataset, img_size, img_size, 1))
    x_data = x_data/255
    
    model.fit(x_data, y_data, epochs=50, batch_size=128, verbose=1, validation_data=(x_data, y_data),callbacks=[callbacks])
    
    scores = model.evaluate(x_data, y_data, verbose=0)
    print(scores)
    
    model_json = model.to_json()
    with open("model-bw.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model-bw.h5')
    
    #return model

model_training()

K.clear_session();


#######################################

