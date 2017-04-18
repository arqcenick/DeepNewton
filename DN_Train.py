import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io


import matplotlib.pyplot as plt
import scipy.misc
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Reshape, merge
from keras.layers.core import Activation, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D, Cropping2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.utils import np_utils
from keras import initializations
from PIL import Image


DATA_PATH = "./data/train_imgs/"


def get_train_list(data_path):
    l = glob.glob(os.path.join(data_path,"*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    print len(l)
    train_list = []
    for f in l:
        if os.path.exists(f):
				train_list.append(f);
    return train_list


def load_train_images(train_list):

    img_list = []
    for img in train_list:
        img_read = input_img = scipy.io.loadmat(img)['img_save']
        img_list.append(img_read)

    return img_list

if __name__ == '__main__':

    train_list = get_train_list(DATA_PATH)
    train_images = load_train_images(train_list)

    adam=Adam(lr=0.0001, beta_1=0.5 )
    im_size = 256


    input_lr = Input(shape=(1, im_size, im_size), dtype='float32', name='encode_input')
    conv1 = Convolution2D(32, 3, 3, border_mode='same')(input_lr)
    act1 = Activation('relu')(conv1)
    conv2 = Convolution2D(32, 3, 3,border_mode='same')(act1)
    act2 = Activation('relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(4, 4))(act2)
    conv3 = Convolution2D(64, 3, 3,border_mode='same')(pool1)
    act3 = Activation('relu')(conv3)
    conv4 = Convolution2D(64, 3, 3,border_mode='same')(act3)
    act4 = Activation('relu')(conv4)
    pool2 = MaxPooling2D(pool_size=(4, 4))(act4)
    conv5 = Convolution2D(128, 3, 3,border_mode='same')(pool2)
    act5 = Activation('relu')(conv5)
    #pool3 = MaxPooling2D(pool_size=(4, 4))(act5)
    conv6 = Convolution2D(256, 3, 3,border_mode='same')(act5)
    act6 = Activation('relu')(conv6)
    #pool4 = MaxPooling2D(pool_size=(4, 4))(act6)
    conv7 = Convolution2D(512, 3, 3,border_mode='same')(act6)
    act7 = Activation('relu')(conv7)
    encoder = Model(input=[input_lr], output=[act7])
    encoder.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    code_lr = Input(shape=(512,4,4), dtype='float32', name='decode_input')
    deconv1 = Convolution2D(512, 3, 3, border_mode='same')(code_lr)
    deact1 = Activation('relu')(deconv1)
    #up1 = UpSampling2D(size=(4,4))(deact1)
    deconv2 = Convolution2D(256, 3, 3, border_mode='same')(deact1)
    deact2 = Activation('relu')(deconv2)
    #up2 = UpSampling2D(size=(4,4))(deact2)
    deconv3 = Convolution2D(128, 3, 3, border_mode='same')(deact2)
    deact3 = Activation('relu')(deconv3)
    up3 = UpSampling2D(size=(4,4))(deact3)
    deconv4 = Convolution2D(64, 3, 3, border_mode='same')(up3)
    deact4 = Activation('relu')(deconv4)
    up4 = UpSampling2D(size=(4,4))(deact4)
    deconv5 = Convolution2D(32, 3, 3, border_mode='same')(up4)
    deact5 = Activation('relu')(deconv5)
    deconv6 = Convolution2D(1, 3, 3, border_mode='same')(deact5)
    deact6 = Activation('relu')(deconv6)
    decoder = Model(input=[code_lr],output=[deact6])
    decoder.compile(optimizer=adam, loss='mse', metrics=['accuracy'])


    codec_input = Input(shape=(1, im_size, im_size), dtype='float32', name='encode_input')
    decoder_input = encoder(codec_input)
    codec_output = decoder(decoder_input)
    codec = Model(input=[codec_input], output=[codec_output]);
    codec.compile(loss='mse', optimizer=adam, metrics=['accuracy'])


    train_images_np = np.reshape(np.asarray(train_images), (-1, 1, 256, 256))

    nbEpoch = 1000
    batchSize = 1
    numBatches = len(train_list)/batchSize

    for epoch in range(1, nbEpoch + 1):
        indices = np.random.permutation(len(train_list));
        for i in range(numBatches):
            randInd = indices[i*batchSize:(i+1)*batchSize]

            print(train_images_np.shape)
            codec_loss = codec.train_on_batch(train_images_np[randInd], train_images_np[randInd])
            print(codec_loss)
        print("Epoch %d" % epoch)
        generated = codec.predict(np.reshape(train_images_np[0,:,:,:], (1,1,256,256)))
    plt.imshow(generated[0,0,:,:], cmap='gray')

    plt.show()
