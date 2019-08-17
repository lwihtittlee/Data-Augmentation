import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import tools as tl

classes     = ['snow','stage','sunset','text','waterfall']
n=2
for index, name in enumerate (classes):

    origin   = 'E:/SCENE/dataset/train/' + name
    result   = 'E:/SCENE/dataset/train/' + name
    x = tf.placeholder(dtype=tf.uint8)
    y = tf.image.rot90(x,k=1)
    with tf.Session()as sess:
        for j in range(2000):
            try:
                img = cv2.imread( origin + '/%05d'%j + '.jpg')

                # shape = img.shape

                # rot_180 = tf.image.rot90(image, k=2)

                output = sess.run(y,feed_dict={x:img})
                i = j+ 2000 * n
                print(i)
                cv2.imwrite( result + '/%05d' %i + '.jpg' , output)
            except:
                print('MIssing')


