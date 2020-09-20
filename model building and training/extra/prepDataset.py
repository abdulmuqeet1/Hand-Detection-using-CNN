# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:03:50 2020

@author: Abdul
"""

import os
import numpy as np
import cv2


lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('C:\\Users\\Abdul\\Desktop\\py_work\\newhmr\\my_work\\leapGestRecog\\00'):
    # to avoid hidden files
    if not j.startswith('.'): 
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1

def get_dataset():
    x_data = []
    y_data = []
    img_size = 150
    datacount = 0 # to tally total images in dataset
    extralist = []
        
    
    for i in range(0,10):
        for j in os.listdir('C:\\Users\Abdul\Desktop\py_work\newhmr\my_work\leapGestRecog\0' + str(i) + '\\'):
            if not j.startswith('.'):
                count = 0 # to tally total images for given specific gesture
                for k in os.listdir('C:\\Users\Abdul\Desktop\py_work\newhmr\my_work\leapGestRecog0'+ str(i)+'\\'+ j+ '\\' ):
                    path = 'C:\\Users\\Abdul\\Desktop\\py_work\\newhmr\\my_work\\leapGestRecog\\0'+ str(i)+'\\'+ str(j)+ '\\'+ str(k)
                    #print(path)
                    extralist.append(path)
                    break
        #             img = cv2.imread("C:\\Users\Abdul\Desktop\py_work\my_work\leapGestRecog\00\01_palm\frame_00_01_0001.png", cv2.IMREAD_GRAYSCALE())
        #             img = cv2.resize(img,  (img_size, img_size))
        #             arr = np.append(img)
        #             x_data.append(arr)
        #             count+=1
        #         y_values = np.full((count, 1), lookup[j])
        #         y_data.append(y_values)
        #         datacount+=1
          
        # x_data = np.array(x_data, dtype='float32')
        # y_data = np.array(y_data)
        # y_data = y_data.reshape(datacount, 1)
        
        #return x_data, y_data
        print(extralist)

                    
get_dataset()
    
img = cv2.imread("C:\\Users\Abdul\Desktop\py_work\my_work\leapGestRecog\00\01_palm\frame_00_01_0001.png")
def get_classes(path):
    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir(path):
        # to avoid hidden files
        if not j.startswith('.'): 
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1
    return lookup
    
get_classes('C:\\Users\\Abdul\\Desktop\\py_work\\newhmr\\my_work\\leapGestRecog\\00')
