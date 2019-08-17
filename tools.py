'''
Popular Augmentation Techniques
'''


import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2


'''Flip'''
def flip(image):
    y = np.fliplr(image)
    return y
'''
# NumPy.'img' = A single image.
flip_1 = np.fliplr(img)
# TensorFlow. 'x' = A placeholder for an image.
shape = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
flip_2 = tf.image.flip_up_down(x)
flip_3 = tf.image.flip_left_right(x)
flip_4 = tf.image.random_flip_up_down(x)
flip_5 = tf.image.random_flip_left_right(x)
'''


'''Rotation'''
# Placeholders: 'x' = A single image, 'y' = A batch of images
# 'k' denotes the number of 90 degree anticlockwise rotations
def rotation(image):
    shape = image.shape

    x = tf.placeholder(dtype=tf.uint8,shape=shape )
    rot_90 = tf.image.rot90(image, k=1)
    #rot_180 = tf.image.rot90(image, k=2)
    with tf.Session()as sess:
        y= sess.run(rot_90,feed_dict={x:image})
    return y

'''
shape = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
rot_90 = tf.image.rot90(img, k=1)
rot_180 = tf.image.rot90(img, k=2)
# To rotate in any angle. In the example below, 'angles' is in radians
shape = [batch, height, width, 3]
y = tf.placeholder(dtype = tf.float32, shape = shape)
rot_tf_180 = tf.contrib.image.rotate(y, angles=3.1415)
# Scikit-Image. 'angle' = Degrees. 'img' = Input Image
# For details about 'mode', checkout the interpolation section below.
rot = skimage.transform.rotate(img, angle=45, mode='reflect')
'''



def scale(image):
    shape = image.shape
    x = tf.placeholder(dtype=tf.float32, shape=shape)

    scale_out = skimage.transform.rescale(x, scale=2.0, mode='constant')     # scale out
    # scale_in = skimage.transform.rescale(x, scale=0.5, mode='constant')    # scale in

    with tf.Session() as sess:
        y = sess.run(scale_out, feed_dict={x: image})
    return y
# '''Scale'''
# # Scikit Image. 'img' = Input Image, 'scale' = Scale factor
# # For details about 'mode', checkout the interpolation section below.
# scale_out = skimage.transform.rescale(img, scale=2.0, mode='constant')
# scale_in = skimage.transform.rescale(img, scale=0.5, mode='constant')
# # Don't forget to crop the images back to the original size (for
# # scale_out)




def crop(image):
    shape = image.shape

    crop_size = [128,128]
    seed = np.random.randint(1234)
    b = tf.random_crop(image, size = crop_size, seed = seed)
    output = tf.image.resize_images(b, size = shape)


    return output
# '''Crop'''
# # TensorFlow. 'x' = A placeholder for an image.
# original_size = [height, width, channels]
# x = tf.placeholder(dtype = tf.float32, shape = original_size)
# # Use the following commands to perform random crops
# crop_size = [new_height, new_width, channels]
# seed = np.random.randint(1234)
# x = tf.random_crop(x, size = crop_size, seed = seed)
# output = tf.images.resize_images(x, size = original_size)



def translation(image):
    shape = image.shape
    a = tf.placeholder(dtype=tf.float32, shape=shape)

    b = tf.image.pad_to_bounding_box( a, 50, 50, 128, 128)
    output = tf.image.crop_to_bounding_box(b, 50, 50, 128, 128)
    with tf.Session() as sess:
        y = sess.run(output, feed_dict={a: image})
    return y

# '''Translation'''
# # pad_left, pad_right, pad_top, pad_bottom denote the pixel
# # displacement. Set one of them to the desired value and rest to 0
# shape = [batch, height, width, channels]
# x = tf.placeholder(dtype = tf.float32, shape = shape)
# # We use two functions to get our desired augmentation
# x = tf.image.pad_to_bounding_box(x, pad_top, pad_left, height + pad_bottom + pad_top, width + pad_right + pad_left)
# output = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, height, width)

def noise(image):
    shape = image.shape
    x = tf.placeholder(dtype=tf.float32, shape=shape)

    noise = tf.random_normal(shape= tf.shape(x),mean=0.0,stddev=1.0,dtype= tf.float32)
    with tf.Session() as sess:
        y = sess.run(noise,feed_dict={x:image})
        return y

# '''Gaussion Noise'''
# #TensorFlow. 'x' = A placeholder for an image.
# shape = [height, width, channels]
# x = tf.placeholder(dtype = tf.float32, shape = shape)
# # Adding Gaussian noise
# noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0,
# dtype=tf.float32)
# output = tf.add(x, noise)
