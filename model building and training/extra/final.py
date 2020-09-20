# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:06:28 2020

@author: Abdul
"""


import numpy as np
from keras.models import model_from_json
#import operator
import cv2
import sys, os
#import matplotlib.pyplot as plt

def load_model(name):
    json_file = open(name, "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model-bw.h5")
    print("Loaded model from disk")
    return loaded_model
    

def check_result(index):
    pass


def make_prediction():
    video_capture = cv2.VideoCapture(0)
    model = load_model("model-bw.json")
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (150, 150))
        gray = np.array(gray, dtype='float32')
        gray = gray.reshape((1, 150, 150, 1))
        
        hand_gesture = model.predict(gray)
    
        location = hand_gesture.argmax()
        
        if location == 0:
            cv2.putText(frame, 'palm', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(255,210,0))
        elif location ==1:
            cv2.putText(frame, 'L', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(255,210,0))
        elif location ==2:
            cv2.putText(frame, 'fist', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(255,210,0))
        elif location ==3:
            cv2.putText(frame, 'thumb', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(255,210,0))
        elif location ==4:
            cv2.putText(frame, 'ok', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(255,210,0))
        elif location ==5:
            cv2.putText(frame, 'C', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(255,210,0))
        else:
            cv2.putText(frame, 'unable to detect', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,0,0))
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

make_prediction()

