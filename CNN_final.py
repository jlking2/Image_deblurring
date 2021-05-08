# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:00:19 2021

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
import time
import matplotlib.pyplot as plt
import imageio
import scipy
import scipy.io
import sys
from scipy import signal
from scipy.fftpack import fftn, ifftn, fftshift

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

## Define Blur function ##
g1 = gaussian_filter(k=5, sigma=3)

fig = 1
plt.figure(fig)
plt.imshow(g1, cmap='hot', interpolation='nearest')
plt.colorbar()

(img_original, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
img_original = img_original/np.max(img_original)
img_blur = np.zeros(np.shape(img_original))

for kk in range(0,np.shape(img_original)[0]):
    img_blur[kk,:,:] = scipy.signal.convolve2d(img_original[kk,:,:], g1, mode='same', boundary='fill', fillvalue=0)

## TURN INTO 0/1D ARRAYS ##
img_original_0D = img_original.reshape(np.shape(img_original)[0],784)
img_blur_0D = img_blur.reshape(np.shape(img_blur)[0],784)

img_original_test = img_original_0D[0:1000,:]
img_original_train = img_original_0D[1000:3000,:]
img_original_validate = img_original_0D[3000:4000,:]

img_blur_test = img_blur_0D[0:1000,:]
img_blur_train = img_blur_0D[1000:3000,:]
img_blur_validate = img_blur_0D[3000:4000,:]

t0 = time.time()
## CONSTRUCT MODEL ##

inputs = keras.Input(shape=(784,))
dense = layers.Dense(784, activation = "relu")
x = dense(inputs)

outputs = layers.Dense(784)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer='Adam',
    metrics=["mean_squared_error"],)

epoch_default = 200
history = model.fit(img_blur_train, img_original_train, batch_size=64, epochs=epoch_default, validation_split=0)
img_recover_test = model.evaluate(img_blur_test, img_original_test, verbose=0)[0]
img_recover_validate = model.evaluate(img_blur_test, img_original_test, verbose=0)[0]
#print("Test loss:", test_imgs[0])
#print("Test accuracy:", test_imgs[1])
## DEPLOY MODEL ##

#img_recover_test = model.predict(img_blur_test)
#img_test_diff = np.mean((img_recover_test - img_original_test)**2)

weights1 = tf.keras.optimizers.Optimizer.weights


## SHOW OUTPUT ##
example = img_original[0,:,:]
example_flat = example.reshape(1,784)

example_blur = img_blur[0,:,:]
example_blur_flat = example_blur.reshape(1,784)

example_output = model.predict(example_blur_flat)
example_output_array = example_output.reshape(28,28)
example_output_array_scale = example_output_array/np.max(example_output_array)

fig += 1
plt.figure(fig)
plt.imshow(example, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

fig += 1
plt.figure(fig)
plt.imshow(example_blur, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

fig += 1
plt.figure(fig)
plt.imshow(example_output_array, cmap="gray", vmin=0, vmax=1)
plt.axis('off')


example = img_original[1,:,:]
example_flat = example.reshape(1,784)

example_blur = img_blur[1,:,:]
example_blur_flat = example_blur.reshape(1,784)

example_output = model.predict(example_blur_flat)
example_output_array = example_output.reshape(28,28)
example_output_array_scale = example_output_array/np.max(example_output_array)

fig += 1
plt.figure(fig)
plt.imshow(example, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

fig += 1
plt.figure(fig)
plt.imshow(example_blur, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

fig += 1
plt.figure(fig)
plt.imshow(example_output_array, cmap="gray", vmin=0, vmax=1)
plt.axis('off')


example = img_original[2,:,:]
example_flat = example.reshape(1,784)

example_blur = img_blur[2,:,:]
example_blur_flat = example_blur.reshape(1,784)

example_output = model.predict(example_blur_flat)
example_output_array = example_output.reshape(28,28)
example_output_array_scale = example_output_array/np.max(example_output_array)

fig += 1
plt.figure(fig)
plt.imshow(example, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

fig += 1
plt.figure(fig)
plt.imshow(example_blur, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

fig += 1
plt.figure(fig)
plt.imshow(example_output_array, cmap="gray", vmin=0, vmax=1)
plt.axis('off')








#fig += 1
#plt.figure(fig)
#plt.imshow(example_output_array_scale, cmap="gray", vmin=0, vmax=1)
#plt.axis('off')

t1 = time.time()
time_loop = t1-t0
sys.exit()
np.savetxt("MNIST_example_clear.csv", example, delimiter=",")
np.savetxt("MNIST_example_blur.csv", example_blur, delimiter=",")
np.savetxt("MNIST_example_recover.csv", example_output, delimiter=",")

scipy.io.savemat('img_original_test.mat', mdict={'img_original_test': img_original_test})
scipy.io.savemat('img_blur_test.mat', mdict={'img_blur_test': img_blur_test})
























import tensorflow as tf
import keras
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/

image_size = 784 # 28*28
epoch_default = 100
nodes_default = 20

## Epoch loop sigmoid
model = Sequential()


# The input layer requires the special input_shape parameter which should match
# the shape of our training data.

#model.add(Dense(units=nodes_default, activation='relu', input_shape=(image_size,)))
#model.add(Dense(units=image_size, activation='relu', input_shape=(image_size,)))
#model.add(Dense(units=num_classes, activation='softmax'))

#model.add(Dense(units=int(nodes_default), activation='relu', input_shape=(image_size,)))
#model.add(Dense(units=int(nodes_default), activation='relu', input_shape=(image_size,)))
#model.add(Dense(units=int(nodes_default), activation='relu', input_shape=(image_size,)))
#model.add(Dense(units=int(image_size), activation='relu', input_shape=(image_size,)))
#model.add(Dense(units=int(nodes_default), activation='sigmoid', input_shape=(image_size,)))
#model.add(Dense(units=int(nodes_default), activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=int(image_size), activation='sigmoid', input_shape=(image_size,)))
#model.add(Dense(units=int(image_size), activation='sigmoid', input_shape=(image_size,)))
#model.add(Dense(units=int(image_size), activation='sigmoid', input_shape=(image_size,)))
#model.add(Dense(units=image_size, activation='softmax'))
#model.add(Dense(units=image_size, activation='relu', input_shape=(image_size,)))
#model.add(Dense(units=nodes_default, activation='sigmoid', input_shape=(nodes_default,)))
#model.add(Dense(units=nodes_default, activation='sigmoid', input_shape=(image_size,)))
#model.add(Dense(units=image_size, activation='sigmoid', input_shape=(image_size,)))

model.summary()
model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['mean_squared_error'])
history = model.fit(img_blur_train, img_original_train, batch_size=128, epochs=epoch_default, verbose=False, validation_split=0)
big_output  = model.evaluate(img_blur_train, img_original_train, verbose=0)
weights1 = tf.keras.optimizers.Optimizer.weights


## SHOW OUTPUT ##
example = img_original[0,:,:]
example_flat = example.reshape(1,784)

example_blur = img_blur[0,:,:]
example_blur_flat = example_blur.reshape(1,784)

example_output = model.predict(example_blur_flat)
example_output_array = example_output.reshape(28,28)
example_output_array_scale = example_output_array/np.max(example_output_array)

x = np.linspace(0, 2 * np.pi, 100)
y = np.cos(x)

scipy.io.savemat('test.mat', dict(img_original, y=y))

#np.savetxt("MNIST_example_clear.csv", example, delimiter=",")
#np.savetxt("MNIST_example_blur.csv", example_blur, delimiter=",")
#np.savetxt("MNIST_example_recover.csv", example_output, delimiter=",")



fig += 1
plt.figure(fig)
plt.imshow(example, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

fig += 1
plt.figure(fig)
plt.imshow(example_blur, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

fig += 1
plt.figure(fig)
plt.imshow(example_output_array, cmap="gray", vmin=0, vmax=1)
plt.axis('off')

fig += 1
plt.figure(fig)
plt.imshow(example_output_array_scale, cmap="gray", vmin=0, vmax=1)
plt.axis('off')


sys.exit()


print(np.max(example_output_reshape))
print(np.max(example_clear))


sys.exit()
sigmoid_epoch_test2L = accuracy
sigmoid_epoch_train2L = history.history['accuracy'][-1]
    
sys.exit()
































# Node loop sigmoid
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=int(nodes_list[kk]), activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='sigmoid', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=epoch_default, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    sigmoid_nodes_test2L[kk] = accuracy
    sigmoid_nodes_train2L[kk] = history.history['accuracy'][-1]

## Epoch loop sigmoid
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=nodes_default, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=nodes_default, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='sigmoid', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=int(epoch_list[kk]), verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    sigmoid_epoch_test3L[kk] = accuracy
    sigmoid_epoch_train3L[kk] = history.history['accuracy'][-1]

# Node loop sigmoid
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=int(nodes_list[kk]), activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=int(nodes_list[kk]), activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='sigmoid', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=epoch_default, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    sigmoid_nodes_test3L[kk] = accuracy
    sigmoid_nodes_train3L[kk] = history.history['accuracy'][-1]




x_train_0D = x_train_0D.astype('float32') / 255
x_test_0D = x_test_0D.astype('float32') / 255




# Epoch loop relu
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=nodes_default, activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='relu', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=int(epoch_list[kk]), verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    relu_epoch_test2L[kk] = accuracy
    relu_epoch_train2L[kk] = history.history['accuracy'][-1]

# Node loop relu
for kk in range(0,len(nodes_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=int(nodes_list[kk]), activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='relu', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=epoch_default, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    relu_nodes_test2L[kk] = accuracy
    relu_nodes_train2L[kk] = history.history['accuracy'][-1]

# Epoch loop relu
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=nodes_default, activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=nodes_default, activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='relu', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=int(epoch_list[kk]), verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    relu_epoch_test3L[kk] = accuracy
    relu_epoch_train3L[kk] = history.history['accuracy'][-1]
    
# Node loop relu
for kk in range(0,len(nodes_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=int(nodes_list[kk]), activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=int(nodes_list[kk]), activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='relu', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=epoch_default, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    relu_nodes_test3L[kk] = accuracy
    relu_nodes_train3L[kk] = history.history['accuracy'][-1]


#plt.figure(1)
#plt.plot(epoch_list, relu_epoch_test2L,marker='o',markersize=12,color='black')
#plt.plot(epoch_list, relu_epoch_test3L,marker='v',markersize=12,color='black')
#plt.plot(epoch_list, sigmoid_epoch_test2L,marker='o',markersize=12,color='red')
#plt.plot(epoch_list, sigmoid_epoch_test3L,marker='v',markersize=12,color='red')
##plt.title('Predicted Labels, changing alpha')
#plt.ylabel('accuracy')
#plt.xlabel('epochs')
#plt.legend(['relu, 2 layers', 'relu, 3 layers', 'sigmoid, 2 layers', 'sigmoid, 3 layers'], loc='best')
#plt.show()
#
#plt.figure(2)
#plt.plot(nodes_list, relu_nodes_test2L,marker='o',markersize=12,color='black')
#plt.plot(nodes_list, relu_nodes_test3L,marker='v',markersize=12,color='black')
#plt.plot(nodes_list, sigmoid_nodes_test2L,marker='o',markersize=12,color='red')
#plt.plot(nodes_list, sigmoid_nodes_test3L,marker='v',markersize=12,color='red')
##plt.title('Predicted Labels, changing alpha')
#plt.ylabel('accuracy')
#plt.xlabel('width of hidden layers')
#plt.legend(['relu, 2 layers', 'relu, 3 layers', 'sigmoid, 2 layers', 'sigmoid, 3 layers'], loc='best')
#plt.show()

plt.figure(3)
plt.plot(epoch_list, 1-relu_epoch_test2L,marker='o',markersize=12,color='black')
plt.plot(epoch_list, 1-relu_epoch_test3L,marker='v',markersize=12,color='black')
plt.plot(epoch_list, 1-sigmoid_epoch_test2L,marker='o',markersize=12,color='red')
plt.plot(epoch_list, 1-sigmoid_epoch_test3L,marker='v',markersize=12,color='red')
#plt.title('Predicted Labels, changing alpha')
plt.ylabel('Error Rate')
plt.xlabel('Epochs')
plt.legend(['relu, 1 layer', 'relu, 2 layers', 'sigmoid, 1 layer', 'sigmoid, 2 layers'], loc='best')
plt.grid(True)
plt.show()

plt.figure(4)
plt.plot(nodes_list, 1-relu_nodes_test2L,marker='o',markersize=12,color='black')
plt.plot(nodes_list, 1-relu_nodes_test3L,marker='v',markersize=12,color='black')
plt.plot(nodes_list, 1-sigmoid_nodes_test2L,marker='o',markersize=12,color='red')
plt.plot(nodes_list, 1-sigmoid_nodes_test3L,marker='v',markersize=12,color='red')
#plt.title('Predicted Labels, changing alpha')
plt.ylabel('Error Rate')
plt.xlabel('Width of hidden layers')
plt.legend(['relu, 1 layer', 'relu, 2 layers', 'sigmoid, 1 layer', 'sigmoid, 2 layers'], loc='best')
plt.grid(True)
plt.show()