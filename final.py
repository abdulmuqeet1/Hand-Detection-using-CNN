# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:06:28 2020

@author: Abdul
"""


import numpy as np
from keras.models import model_from_json
import cv2
import sys, os
import time

def load_model(name):
    json_file = open(name, "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model-bw.h5")
    #print("Loaded model from disk")
    return loaded_model


def make_prediction():
    video_capture = cv2.VideoCapture(0)
    model = load_model("model-bw.json")
    while True:
        ret, frame = video_capture.read()
        frame = cv2.rectangle(frame, (400,30), (600,300), color=(255,0,0), thickness=1)
        #frame[y1:y2, x1:x2]
        img = frame[35:295, 405:595]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (150, 150))
        img = np.array(img, dtype='float32')
        img = img.reshape((1, 150, 150, 1))

        hand_gesture = model.predict(img)

        location = hand_gesture.argmax()

        if location == 0:
            cv2.putText(frame, 'Fist', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0,255,0), thickness = 2)
        elif location ==1:
            cv2.putText(frame, 'L', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0,255,0), thickness = 2)
        elif location ==2:
            cv2.putText(frame, 'Ok', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0,255,0), thickness = 2)
        elif location ==3:
            cv2.putText(frame, 'Palm', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0,255,0), thickness = 2)
        elif location ==4:
            cv2.putText(frame, 'Thumb up', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0,255,0), thickness = 2)
        elif location ==5:
            cv2.putText(frame, 'victory', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0,255,0), thickness = 2)
        else:
            cv2.putText(frame, 'unable to detect', org=(10,480), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,0,255))

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)

    video_capture.release()
    cv2.destroyAllWindows
    return "Thank You"

#make_prediction()
