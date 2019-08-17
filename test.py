import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt

img = cv2.imread('1.jpg')
# print(img.shape)
print(img.dtype)

## flip

# NumPy.'img' = A single image.
flip_1 = np.fliplr(img)

# TensorFlow. 'x' = A placeholder for an image.
shape = img.shape
print(shape)
x = tf.placeholder(dtype= np.uint8 ,shape = shape)
flip_3 = tf.image.flip_left_right(x)
with tf.Session() as sess:
    flip_3_output = sess.run(flip_3,feed_dict={x:img})

cv2.imshow('s',flip_1)
cv2.imshow('a',flip_3_output)

cv2.imwrite('a.jpg',flip_1)
cv2.waitKey(0)