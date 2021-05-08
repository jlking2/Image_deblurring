# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:41:53 2021

@author: jlking2
"""
def gaussian_filter(k, sigma):
    ''' Gaussian filter
    :param k: defines the lateral size of the kernel/filter, default 5
    :param sigma: standard deviation (dispersion) of the Gaussian distribution
    :return matrix with a filter [k x k] to be used in convolution operations
    '''
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    return filt / np.sum(filt)

import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy
import sys
import time
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from numpy import random
from scipy import signal
from scipy.fftpack import fftn, ifftn, fftshift

(img_original, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
img_original = img_original/np.max(img_original)

img_original_test = img_original[0:1000,:,:]
img_original_train = img_original[1000:3000,:,:]
img_original_validate = img_original[3000:4000,:,:]

img_original_train_flat = img_original_train.reshape(np.shape(img_original_train)[0],784)
img_original_test_flat = img_original_test.reshape(np.shape(img_original_test)[0],784)
img_original_validate_flat = img_original_validate.reshape(np.shape(img_original_validate)[0],784)

## Define Blur function ##
g1 = gaussian_filter(k=5, sigma=3)
plt.figure(1)
plt.imshow(g1, cmap='hot', interpolation='nearest')
plt.colorbar()

img_blur = np.zeros(np.shape(img_original))
for kk in range(0,np.shape(img_original)[0]):
    img_blur[kk,:,:] = scipy.signal.convolve2d(img_original[kk,:,:], g1, mode='same', boundary='fill', fillvalue=0)

img_blur_test = img_blur[0:1000,:,:]
img_blur_train = img_blur[1000:3000,:,:]
img_blur_validate = img_blur[3000:4000,:,:]

img_blur_test_flat = img_blur_test.reshape(np.shape(img_blur_test)[0],784)
img_blur_train_flat = img_blur_train.reshape(np.shape(img_blur_train)[0],784)
img_blur_validate_flat = img_blur_validate.reshape(np.shape(img_blur_validate)[0],784)

lambda_set = np.logspace(-14, 2, num=17)
lambda_plot_num = 9
img_test_diff_array = np.zeros((np.shape(lambda_set)))
img_validate_diff_array = np.zeros((np.shape(lambda_set)))
t0 = time.time()
for qq in range(0,np.shape(lambda_set)[0]):
    lambda_1 = lambda_set[qq]
    A_width = np.shape(img_blur_train_flat)[1]
    eyeye = lambda_1*np.eye(A_width)
    
    t1 = time.time()
    t_pix = t1-t0
    
    ## Whole image looping
    
    w_hat_series = np.zeros((np.shape(img_original_train_flat)[1], np.shape(img_original_train_flat)[1]))
    for rr in range(0,np.shape(img_original_train_flat)[1]):
        img_pix = img_original_train_flat[:,rr]
        w_hat = np.linalg.inv(img_blur_train_flat.T@img_blur_train_flat+eyeye)@img_blur_train_flat.T@img_pix
        w_hat_series[:,rr] = w_hat
        
    t2 = time.time()
    t_loop = t2-t1
    #sys.exit()
      
    img_recover_test_flat = np.zeros((np.shape(img_original_test_flat)))
    for rr in range(0,np.shape(img_original_test_flat)[0]):
        #print(rr)
        img_recov = img_blur_test_flat[rr,:]@w_hat_series
        img_recover_test_flat[rr,:] = img_recov
    img_recover_test = img_recover_test_flat.reshape(np.shape(img_original_test_flat)[0],28,28)
    
    #sys.exit()
    img_recover_validate_flat = np.zeros((np.shape(img_original_validate_flat)))
    for rr in range(0,np.shape(img_original_validate_flat)[0]):
        img_recov = img_blur_validate_flat[rr,:]@w_hat_series
        img_recover_validate_flat[rr,:] = img_recov
    img_recover_validate = img_recover_validate_flat.reshape(np.shape(img_original_validate_flat)[0],28,28)
    
    img_test_diff = np.mean((img_recover_test - img_original_test)**2)
    #img_test_diff2 = np.mean((img_recover_test_flat - img_original_test_flat)**2)
    
    img_validate_diff = np.mean((img_recover_validate - img_original_validate)**2)
    #img_validate_diff2 = np.mean((img_recover_validate_flat - img_original_validate_flat)**2)
    
    img_test_diff_array[qq] = img_test_diff
    img_validate_diff_array[qq] = img_validate_diff
    print(lambda_1)
    
    if qq == lambda_plot_num:
        w_keep = w_hat_series
    if qq == lambda_plot_num-1:
        w_keep2 = w_hat_series   
    if qq == lambda_plot_num+1:
        w_keep3 = w_hat_series
            
t3 = time.time()
t_big_loop = t3-t0

lambda_set_abridge = lambda_set[7:15]
img_test_diff_array_abridge = img_test_diff_array[7:15]
img_validate_diff_array_abridge = img_validate_diff_array[7:15]

plot_num = 2
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

plot_num += 1 
plt.figure(plot_num)
plt.plot(lambda_set, img_validate_diff_array, 'bo')
plt.xlabel('Tuning Parameter, \u03BB')
plt.ylabel('MSPE')
plt.xscale('log')
plt.yscale('log')
#plt.title('Error rate vs Ridge Parameter, \u03BB')
plt.grid(True)
plt.show()

plot_num += 1 
plt.figure(plot_num)
plt.plot(lambda_set_abridge, img_validate_diff_array_abridge, 'bo')
plt.xlabel('Tuning Parameter, \u03BB')
plt.ylabel('MSPE')
plt.xscale('log')
plt.yscale('log')
plt.ylim([1E-5, 1E-2])
#plt.title('Error rate vs Ridge Parameter, \u03BB')
plt.grid(True)
plt.show()

plt.figure(plot_num)
plt.rc('font', **font)
plt.plot(lambda_set, img_test_diff_array, 'bo')
plt.xlabel('Tuning Parameter, \u03BB')
plt.ylabel('MSPE')
plt.xscale('log')
plt.yscale('log')
#plt.title('Error rate vs Ridge Parameter, \u03BB')
plt.grid(True)
plt.show()

plot_num += 1 
plt.figure(plot_num)
plt.plot(lambda_set_abridge, img_test_diff_array_abridge, 'bo')
plt.xlabel('Tuning Parameter, \u03BB')
plt.ylabel('MSPE')
plt.xscale('log')
plt.yscale('log')
plt.ylim([1E-5, 1E-2])
#plt.title('Error rate vs Ridge Parameter, \u03BB')
plt.grid(True)
plt.show()

example = img_original[0,:,:]
example_flat = example.reshape(1,784)

example_blur = img_blur_test[0,:,:]
example_blur_flat = example_blur.reshape(1,784)

d_test = example_blur_flat@w_keep
d_test_array = d_test.reshape(28,28)

plot_num += 1 
plt.figure(plot_num)
plt.imshow(example, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

plot_num += 1 
plt.figure(plot_num)
plt.imshow(example_blur, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

plot_num += 1 
plt.figure(plot_num)
plt.imshow(d_test_array, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

example = img_original[1,:,:]
example_flat = example.reshape(1,784)

example_blur = img_blur_test[1,:,:]
example_blur_flat = example_blur.reshape(1,784)

d_test = example_blur_flat@w_keep
d_test_array = d_test.reshape(28,28)

plot_num += 1 
plt.figure(plot_num)
plt.imshow(example, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

plot_num += 1 
plt.figure(plot_num)
plt.imshow(example_blur, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

plot_num += 1 
plt.figure(plot_num)
plt.imshow(d_test_array, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

example = img_original[2,:,:]
example_flat = example.reshape(1,784)

example_blur = img_blur_test[2,:,:]
example_blur_flat = example_blur.reshape(1,784)

d_test = example_blur_flat@w_keep
d_test_array = d_test.reshape(28,28)

plot_num += 1 
plt.figure(plot_num)
plt.imshow(example, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

plot_num += 1 
plt.figure(plot_num)
plt.imshow(example_blur, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

plot_num += 1 
plt.figure(plot_num)
plt.imshow(d_test_array, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

##OPTIMAL 
#Optimal_tuning = 1E-6






