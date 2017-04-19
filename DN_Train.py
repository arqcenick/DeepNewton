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
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D, Cropping2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.utils import np_utils
from PIL import Image
from natsort import natsorted


DATA_PATH = "./data/train_imgs/"
TEST_PATH = "./data/test_imgs/"


def get_image_list(data_path):
    l = glob.glob(os.path.join(data_path,"*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    l = natsorted(l)
    print len(l)
    train_list = []
    for f in l:
        if os.path.exists(f):
				train_list.append(f);
    return train_list


def load_images(image_list):

    img_list = []
    for img in image_list:
        img_read = input_img = scipy.io.loadmat(img)['img_save']
        img_list.append(img_read)

    return img_list

if __name__ == '__main__':

    train_list = get_image_list(DATA_PATH)
    train_images = load_images(train_list)
    test_list = get_image_list(TEST_PATH)
    test_images = load_images(test_list)

    adam=Adam(lr=0.0001, beta_1=0.5 )
    im_size = 128


    input_lr = Input(shape=(1, im_size, im_size), dtype='float32', name='encode_input')
    conv1 = Convolution2D(4, 3, 3, border_mode='same')(input_lr)
    act1 = Activation('relu')(conv1)
    #conv2 = Convolution2D(32, 3, 3,border_mode='same')(act1)
    #act2 = Activation('relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act1)
    conv3 = Convolution2D(8, 3, 3,border_mode='same')(pool1)
    act3 = Activation('relu')(conv3)
    #conv4 = Convolution2D(64, 3, 3,border_mode='same')(act3)
    #act4 = Activation('relu')(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act3)
    conv5 = Convolution2D(16, 3, 3,border_mode='same')(pool2)
    act5 = Activation('relu')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(act5)
    conv6 = Convolution2D(16, 3, 3,border_mode='same')(pool3)
    act6 = Activation('relu')(conv6)
    pool4 = MaxPooling2D(pool_size=(2, 2))(act6)
    conv7 = Convolution2D(32, 3, 3,border_mode='same')(pool4)
    act7 = Activation('relu')(conv7)
    pool5 = MaxPooling2D(pool_size=(2, 2))(act7)
    conv8 = Convolution2D(32, 3, 3,border_mode='same')(pool5)
    act8 = Activation('relu')(conv8)
    flt1 = Flatten()(act8)
    fc1 = Dense(128, activation='relu')(flt1)
    print(fc1.get_shape())
    encoder = Model(input=[input_lr], output=[fc1])
    encoder.compile(optimizer=adam, loss='mse', metrics=['accuracy'])



    code_lr = Input(shape=(128,), dtype='float32', name='decode_input')
    #deconv0 = Convolution2D(512, 3, 3, border_mode='same')(code_lr)
    #deact0 = Activation('relu')(deconv0)
    fc2 = Dense(32*4*4, activation='relu')(code_lr)
    rshp1 = Reshape((32,4,4))(fc2)
    up0 = UpSampling2D(size=(2,2))(rshp1)
    deconv1 = Convolution2D(32, 3, 3, border_mode='same')(up0)
    deact1 = Activation('relu')(deconv1)
    up1 = UpSampling2D(size=(2,2))(deact1)
    deconv2 = Convolution2D(16, 3, 3, border_mode='same')(up1)
    deact2 = Activation('relu')(deconv2)
    up2 = UpSampling2D(size=(2,2))(deact2)
    deconv3 = Convolution2D(16, 3, 3, border_mode='same')(up2)
    deact3 = Activation('relu')(deconv3)
    up3 = UpSampling2D(size=(2,2))(deact3)
    deconv4 = Convolution2D(8, 3, 3, border_mode='same')(up3)
    deact4 = Activation('relu')(deconv4)
    up4 = UpSampling2D(size=(2,2))(deact4)
    deconv5 = Convolution2D(4, 3, 3, border_mode='same')(up4)
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


    train_images_np = np.reshape(np.asarray(train_images), (-1, 1, im_size, im_size))
    test_images_np = np.reshape(np.asarray(test_images), (-1, 1, im_size, im_size))

    nbEpoch = 100
    batchSize = 1
    numBatches = len(train_list)/batchSize
    train_or_load = False
    if train_or_load:
        for epoch in range(1, nbEpoch + 1):
            indices = np.random.permutation(len(train_list));
            for i in range(numBatches):
                randInd = indices[i*batchSize:(i+1)*batchSize]

                #print(train_images_np.shape)
                codec_loss = codec.train_on_batch(train_images_np[randInd], train_images_np[randInd])
                #print(codec_loss, "Epoch",  float(i)/numBatches+(epoch-1))
            print("Epoch %d" % epoch)
            train_error = np.mean(np.square(codec.predict(train_images_np)-train_images_np))
            print("Train Error %s" % train_error)
        enc_model_json = encoder.to_json()
        with open("encoder.json", "w") as json_file:
            json_file.write(enc_model_json)
        # serialize weights to HDF5
        encoder.save_weights("encoder.h5")

        dec_model_json = decoder.to_json()
        with open("decoder.json", "w") as json_file:
            json_file.write(dec_model_json)
        # serialize weights to HDF5
        decoder.save_weights("decoder.h5")

        codec_model_json = codec.to_json()
        with open("codec.json", "w") as json_file:
            json_file.write(codec_model_json)
        # serialize weights to HDF5
        codec.save_weights("codec.h5")
        print("Saved model into h5 file")

        for i in range(len(train_list)):

            generated = codec.predict(np.reshape((train_images_np[i,:,:,:]), (1,1,im_size,im_size)))
            plt.imshow(generated[0,0,:,:], cmap='gray')
            plt.show()
            plt.imshow(train_images_np[i,0,:,:], cmap='gray')
            plt.show()
    else:
        codec.load_weights("codec.h5")
        print("Loaded model from h5 file")






    #flat_input = Input(shape=(1, im_size, im_size))
    #flat_output = encoder(flat_input)
    #flat3 = Flatten()(flat_output)
    #flattener = Model(input=[flat_input], output=[flat3])
    #flattener.compile(optimizer=adam, loss='mse')
    train_codes =  encoder.predict((train_images_np))
    test_codes = encoder.predict((test_images_np))

    print(train_codes.shape)

    #THE LSTM Part

    encode_size = 128

    coded_rnn_input = code_lr = Input(shape=(8,encode_size), dtype='float32', name='lstm_input')
    lstm1 = LSTM(512, return_sequences=True, unroll=True)(coded_rnn_input)
    lstm2 = LSTM(128, unroll=True)(lstm1)
    #print(lstm1.get_shape())
    #dense1 = L(encode_size, activation='relu')(lstm2)
    #print(dense1.get_shape())
    physics = Model(input=[coded_rnn_input], output=[lstm2]);
    physics.compile(loss='mse', optimizer=adam, metrics=['accuracy'])


    def plottrain(past, present, future):
        fig, axes = plt.subplots(1, 9)
        for i, ax in enumerate(axes.flat):
            if i < 4:
                ax.imshow(past[i,0,:,:], cmap='gray')
            elif i == 4:
                ax.imshow(present[0,0,:,:], cmap='gray')
            else:
                #print(i)
                ax.imshow(future[i-5,0,:,:], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()



    timeSize = 8
    train_or_load_lstm = True;
    if train_or_load_lstm:
        nbEpoch = 500

        numtimes = len(train_list)-timeSize-1
        for epoch in range(1, nbEpoch + 1):
            indices = np.random.permutation(len(train_list));
            randperm = np.random.permutation(numtimes)
            for i in range(3, numtimes-1):
                t = randperm[i-2]
                lstm_loss = physics.train_on_batch(np.reshape(train_codes[t:t+timeSize], (1,timeSize,encode_size)), np.reshape(train_codes[t+timeSize:t+timeSize+1], (1,encode_size)))
            print(lstm_loss, "Epoch",  float(i)/numtimes+(epoch-1))

        physics_model_json = encoder.to_json()
        with open("physics.json", "w") as json_file:
            json_file.write(physics_model_json)
        # serialize weights to HDF5
        physics.save_weights("physics.h5")
    else:
        physics.load_weights("physics.h5")
        print("Loaded model from h5 file")
        '''
        reshape_input = Input(shape=(encode_size,))
        reshaper1 = Reshape((64,4,4))(reshape_input)
        decode_output = decoder(reshaper1)
        physics_decode = Model(input=[reshape_input], output=[decode_output])
        physics_decode.compile(optimizer=adam, loss='mse')
        '''
        start = 10
        generate_codes = np.reshape(test_codes[start:start+timeSize], (1,timeSize,encode_size))
        print(generate_codes.shape)
        past = np.reshape(test_images_np[5:5+timeSize], (4, 1, im_size, im_size))
        for j in range(0, len(test_list)-timeSize- start):

            index = np.random.randint(3, len(test_list))
            index = j+start
            generate_codes = np.reshape(test_codes[index:index+timeSize], (1,timeSize,encode_size))
            #print("test code shape", test_codes.shape)
            physics_output = physics.predict(generate_codes)

            for t in range(timeSize-1):
                generate_codes[0,t,:] = generate_codes[0,t+1,:];
            generate_codes[0,timeSize-1,:] = physics_output
            #Reshaper


            guessed = decoder.predict(physics_output)
            for t in range(4-1):
            #    print(t)
                past[t,0,:,:] = past[t+1,0,:,:];
            past[3,0,:,:] = guessed
            #print(guessed.shape)
            #print(test_images_np[j+4,0,:,:].shape)

            concat_guess = np.concatenate((np.squeeze(guessed), test_images_np[j+4+start,0,:,:]), axis=1)
            im = Image.fromarray((concat_guess*255).astype(np.uint8))
            filename = 'predictions/future%d.png' % j
            #print(filename)
            im.save(filename)


            #A = raw_input()
            #plottrain(past, guessed, train_images_np[index+5:index+9])
