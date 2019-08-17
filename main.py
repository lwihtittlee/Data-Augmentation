import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import tools as tl

classes     = ['beach','cat','dog','fireworks','flower','food','greenery','group','night',
               'portrait','sky','snow','stage','sunset','text','waterfall']

def flip():
    n = 1
    for index, name in enumerate (classes):
        origin   = 'E:/SCENE/dataset/train/' + name
        result   = 'E:/SCENE/dataset/train/' + name
        for j in range(2000):
            try:
                img = cv2.imread( origin + '/%05d'%j + '.jpg')
                print(img.shape)
                output = tl.flip(img)
                i = j+2000 * n
                cv2.imwrite( result + '/%05d' %i + '.jpg' , output)
            except:
                print('MIssing')

def rotation():
    n = 2
    for index, name in enumerate(classes):

        origin = 'E:/SCENE/dataset/train/' + name
        result = 'E:/SCENE/dataset/train/' + name

        for j in range(2000):
            try:
                img = cv2.imread(origin + '/%05d' % j + '.jpg')
                print(img.dtype)
                output = tl.rotation(img)
                i = j + 2000 * n
                cv2.imwrite(result + '/%05d' % i + '.jpg', output)
            except:
                print('MIssing')



# flip()
rotation()