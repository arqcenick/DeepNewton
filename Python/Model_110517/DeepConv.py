import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io


import matplotlib.pyplot as plt
import scipy.misc
import keras
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Lambda
from keras.layers import Reshape
#from keras.layers import add
from keras.layers.core import Activation
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Deconvolution2D, Cropping2D, ZeroPadding2D
from keras.layers.merge import Add, Multiply
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.utils import np_utils
from natsort import natsorted
import cv2

DATA_PATH = "./data/multi_train_imgs/"
TEST_PATH = "./data/multi_train_imgs/"


def get_image_list(data_path):
    l = glob.glob(os.path.join(data_path,"*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    l = natsorted(l)
    print len(l)
    train_list = []
    for f in l:
        if os.path.exists(f):
				train_list.append(f)
    return train_list


def load_images(image_list):

    img_list = []
    for img in image_list:
        img_read = input_img = scipy.io.loadmat(img)['img_save']
        img_list.append(img_read)

    return img_list
'''
def add_gaussian_noise(images):
    for img in images:
        noise = np.zeros(128, 128)
        img = img + cv2.randn(img, (0), (99))
'''

if __name__ == '__main__':

    train_list = get_image_list(DATA_PATH)
    train_images = load_images(train_list)
    test_list = get_image_list(TEST_PATH)
    test_images = load_images(test_list)

    adam=Adam(lr=0.0001, beta_1=0.5 )
    im_size = 128


    input_lr = Input(shape=(im_size, im_size, 1), dtype='float32', name='encode_input')
    #gaussian = GaussianNoise(0.1)(input_lr)
    conv1 = Conv2D(2, (3, 3), padding='same')(input_lr)
    print(conv1.get_shape())
    act1 = Activation('relu')(conv1)
    #act2 = Activation('relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act1)
    conv3 = Conv2D(4, (3, 3), padding='same')(pool1)
    act3 = Activation('relu')(conv3)
    #conv4 = Convolution2D(64, 3, 3,border_mode='same')(act3)
    #act4 = Activation('relu')(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act3)
    conv5 = Conv2D(8, (3, 3), padding='same')(pool2)
    act5 = Activation('relu')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(act5)
    conv6 = Conv2D(8, (3, 3), padding='same')(pool3)
    act6 = Activation('relu')(conv6)
    pool4 = MaxPooling2D(pool_size=(2, 2))(act6)
    conv7 = Conv2D(16, (3, 3), padding='same')(pool4)
    act7 = Activation('tanh')(conv7)
    #pool5 = MaxPooling2D(pool_size=(2, 2))(act7)
    #conv8 = Convolution2D(32, 3, 3,border_mode='same')(pool5)
    #act8 = Activation('relu')(conv8)
    print(act7.get_shape())
    encoder = Model(input=[input_lr], output=[act7])
    encoder.compile(optimizer=adam, loss='mse', metrics=['accuracy'])



    code_lr = Input(shape=(8,8,16), dtype='float32', name='decode_input')
    print(code_lr.get_shape())
    #dgaussian = GaussianNoise(0.1)(code_lr)

    #deconv0 = Convolution2D(512, 3, 3, border_mode='same')(code_lr)
    #deact0 = Activation('relu')(deconv0)
    #dfc1 = Dense(32*4*4, activation='relu')(code_lr)
    #dfc2 = Dense(16*8*8, activation='relu')(dgaussian)
    #rshp1 = Reshape((16,8,8))(dfc2)
    #up0 = UpSampling2D(size=(2,2))(rshp1)
    #deconv1 = Convolution2D(32, 3, 3, border_mode='same')(up0)
    #deact1 = Activation('relu')(deconv1)

    up1 = UpSampling2D(size=(2,2))(code_lr)
    deconv2 = Conv2D(8, (3, 3), border_mode='same')(up1)
    deact2 = Activation('relu')(deconv2)
    up2 = UpSampling2D(size=(2,2))(deact2)
    deconv3 = Conv2D(8, (3, 3), border_mode='same')(up2)
    deact3 = Activation('relu')(deconv3)
    up3 = UpSampling2D(size=(2,2))(deact3)
    deconv4 = Conv2D(4, (3, 3), border_mode='same')(up3)
    deact4 = Activation('relu')(deconv4)
    up4 = UpSampling2D(size=(2,2))(deact4)
    deconv5 = Conv2D(2, (3, 3), border_mode='same')(up4)
    deact5 = Activation('relu')(deconv5)
    deconv6 = Conv2D(1, (3, 3), border_mode='same')(deact5)
    deact6 = Activation('relu')(deconv6)
    
    decoder = Model(input=[code_lr],output=[deact6])
    decoder.compile(optimizer=adam, loss='mse', metrics=['accuracy'])


    codec_input = Input(shape=(im_size, im_size, 1), dtype='float32', name='encode_input')
    codec_input_next = Input(shape=(im_size, im_size, 1), dtype='float32', name='encode_input_next')
    decoder_input = encoder(codec_input)
    decoder_input_next = encoder(codec_input_next)
    codec_output = decoder(decoder_input)
    negated = Lambda(lambda x: -x)(decoder_input_next)
    code_diff = Add()([decoder_input, negated])
    codec = Model(input=[codec_input, codec_input_next], output=[codec_output, code_diff])
    codec.compile(loss='mse', optimizer=adam)


    codec_only=Model(input=[codec_input], output=[codec_output])
    codec_only.compile(loss='mse', optimizer=adam)


    train_images_np = np.reshape(np.asarray(train_images), (-1, im_size, im_size, 1))
    test_images_np = np.reshape(np.asarray(test_images), (-1, im_size, im_size, 1))

    nbEpoch = 50
    batchSize = 1
    numBatches = len(train_list)/batchSize-1
    train_or_load = False
    if train_or_load:
        codec.load_weights("codec.h5")
        for epoch in range(1, nbEpoch + 1):
            indices = np.random.permutation(len(train_list)-1)
            for i in range(numBatches):
                randInd = indices[i*batchSize:(i+1)*batchSize]
                #print(randInd)
                #print(train_images_np.shape)
                diff_label = np.zeros((1,8,8,16))
                codec_loss = codec.train_on_batch([train_images_np[randInd], train_images_np[randInd+1]], [train_images_np[randInd], diff_label])
                #print(codec_loss, "Epoch",  float(i)/numBatches+(epoch-1))
            print("Epoch %d" % epoch)
            #print(len(codec_only.predict(train_images_np)))
            train_error = np.mean(np.square(codec_only.predict(train_images_np)-train_images_np))
            print("Train Error %s" % train_error)
            print("Code Diff %s" % codec_loss[1])
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
        codec_only_model_json = codec_only.to_json()
        with open("codec.json", "w") as json_file:
            json_file.write(codec_model_json)
        with open("codec_only.json", "w") as json_file:
            json_file.write(codec_only_model_json)
        # serialize weights to HDF5
        codec.save_weights("codec.h5")
        codec_only.save_weights("codec_only.h5")
        print("Saved model into h5 file")
        
        for i in range(len(train_list)):

            generated = codec_only.predict(np.reshape((train_images_np[i,:,:,:]), (1,im_size,im_size,1)))
            plt.imshow(generated[0,:,:,0], cmap='gray')
            plt.show()
            plt.imshow(train_images_np[i,:,:,0], cmap='gray')
            plt.show()
        
    else:
        codec.load_weights("codec.h5")
        encoder.load_weights("encoder.h5")
        decoder.load_weights("decoder.h5")
        print("Loaded model from h5 file")

    
    




    #flat_input = Input(shape=(1, im_size, im_size))
    #flat_output = encoder(flat_input)
    #flat3 = Flatten()(flat_output)
    #flattener = Model(input=[flat_input], output=[flat3])
    #flattener.compile(optimizer=adam, loss='mse')
    train_codes =  encoder.predict((train_images_np))
    test_codes = encoder.predict((test_images_np))

    print("train_codes shape: ", train_codes.shape)

    #THE Recurrent Part
    codec_input=Input(shape=(8, 8, 16), dtype='float32', name='encode_input')
    hidden_input=Input(shape=(8, 8, 16), dtype='float32', name='hidden_state')
    input_conv  = Conv2D(16,(3,3), padding='same')(codec_input)
    hidden_conv =Conv2D(16,(3,3), padding='same')(hidden_input)
    combine_layer = Add()([input_conv, hidden_conv])
    combine_act = Activation('tanh')(combine_layer)
   
    decay_const = K.constant(0.5, dtype='float32')
    write_const = K.constant(0.5, dtype='float32')
    decay_layer = Lambda(lambda x: decay_const*x)(hidden_input)
    write_layer = Lambda(lambda x: write_const*x)(combine_act)
    update_layer = Add()([decay_layer, write_layer])
    interaction_layer = Conv2D(16,(3,3), padding='same')(update_layer)
    output_layer = Activation('tanh')(update_layer)
    physics_cell = Model(inputs=[codec_input, hidden_input], outputs=[output_layer, interaction_layer])
    physics_cell.compile(loss='mse', optimizer=adam)

    

    codec_input_t1 = Input(shape=(8, 8, 16), dtype='float32', name='encode_input_t1')
    codec_input_t2 = Input(shape=(8, 8, 16), dtype='float32', name='encode_input_t2')
    codec_input_t3 = Input(shape=(8, 8, 16), dtype='float32', name='encode_input_t3')
    codec_input_t4 = Input(shape=(8, 8, 16), dtype='float32', name='encode_input_t4')

    encoder_t1 = Input(shape=(128, 128, 1), dtype='float32', name='encoder_t1')
    encoder_t2 = Input(shape=(128, 128, 1), dtype='float32', name='encoder_t2')
    encoder_t3 = Input(shape=(128, 128, 1), dtype='float32', name='encoder_t3')
    encoder_t4 = Input(shape=(128, 128, 1), dtype='float32', name='encoder_t4')
    

    #is_stateful = K.placeholder(dtype=bool)

    '''
    
    if(is_stateful):
        hidden1 = Input(shape=(1, 8, 8), dtype='float32', name='encode_input_t1')
    else:
    
    
   
    def update_add(x, y):
        return K.update_add(x , y)
    def update_add_output_shape(input_shape):
        return input_shape
    





    print(codec_input_t1.get_shape())
    hidden1 = Input(shape=(8, 8, 32), dtype='float32', name='init_state')
    #hidden1_conv = Conv2D(32,(3,3), padding='same', activation='tanh')(hidden1)
    rnn_conv1 = Conv2D(32,(3,3), padding='same')(codec_input_t1)
    rnn_convh1 = Conv2D(32,(3,3), padding='same')(hidden1)
    addLayer1 = Add()([rnn_conv1, rnn_convh1])
    rnn_act1 =  Activation('tanh')(addLayer1)
    
    
    hidden2 = Add()([rnn_act1, hidden1])
    #hidden2_conv = Conv2D(32,(3,3), padding='same', activation='tanh')(hidden2)
    rnn_conv2 = Conv2D(32,(3,3), padding='same')(codec_input_t2)
    rnn_convh2 = Conv2D(32,(3,3), padding='same')(hidden2)
    addLayer2 = Add()([rnn_conv2, rnn_convh2])
    rnn_act2 =  Activation('tanh')(addLayer2)
    
    
    hidden3 = Add()([rnn_act2, hidden2])
    #hidden3_conv = Conv2D(32,(3,3), padding='same', activation='tanh')(hidden3)
    rnn_conv3 = Conv2D(32,(3,3), padding='same')(codec_input_t3)
    rnn_convh3 = Conv2D(32,(3,3), padding='same')(hidden3)
    addLayer3 = Add()([rnn_conv3, rnn_convh3])
    rnn_act3 =  Activation('tanh')(addLayer3)
    
    
    hidden4 = Add()([rnn_act3, hidden3])
    #hidden4_conv = Conv2D(32,(3,3), padding='same', activation='tanh')(hidden4)
    rnn_conv4 = Conv2D(32,(3,3), padding='same')(codec_input_t4)
    rnn_convh4 = Conv2D(32,(3,3), padding='same')(hidden4)
    addLayer4 = Add()([rnn_conv4, rnn_convh4])
    rnn_act4 =  Activation('tanh')(addLayer4)

    hidden5  = Add()([rnn_act4, hidden4])
    physics_output = Activation('relu')(hidden5)
    #physics_output = Conv2D(32,(3,3), padding='same', activation='relu')(hidden5)
    '''
    

    encode_out_1 = encoder(encoder_t1)
    encode_out_2 = encoder(encoder_t2)
    encode_out_3 = encoder(encoder_t3)
    encode_out_4 = encoder(encoder_t4)

    initial_state = Input(shape=(8, 8, 16), dtype='float32', name='init_state')
    
    [out1, hidden1] = physics_cell([encode_out_1, initial_state])
    diff_layer1 = Add()([out1, encode_out_1])
    [out2, hidden2] = physics_cell([encode_out_2, hidden1])
    diff_layer2 = Add()([out2, encode_out_2])
    [out3, hidden3] = physics_cell([encode_out_3, hidden2])
    diff_layer3 = Add()([out3, encode_out_3])
    [out4, hidden4] = physics_cell([encode_out_4, hidden3])
    diff_layer4 = Add()([out4, encode_out_4])
    
    [out5, hidden5] = physics_cell([diff_layer4, hidden4])
    diff_layer5 = Add()([out5, diff_layer4])
    [out6, hidden6] = physics_cell([diff_layer5, hidden5])
    diff_layer6 = Add()([out6, diff_layer5])
    [out7, hidden7] = physics_cell([diff_layer6, hidden6])
    diff_layer7 = Add()([out7, diff_layer6])
    [out8, hidden8] = physics_cell([diff_layer7, hidden7])
    diff_layer8 = Add()([out8, diff_layer7])
    decoder_out = decoder(diff_layer8)
    
    
    adamlstm=Adam(lr=0.0001, beta_1=0.9 )
    rmsprop = RMSprop(lr=0.0001)
    physics = Model(inputs=[initial_state, encoder_t1, encoder_t2, encoder_t3, encoder_t4], outputs=[decoder_out])
    physics.compile(loss='mse', optimizer=rmsprop)


    #THE LSTM Part

    encode_size = 128
    timeSize = 4
    '''
    coded_rnn_input = code_lr = Input(shape=(timeSize,encode_size), dtype='float32', name='lstm_input')
    lstm1 = LSTM(128, return_sequences=True, unroll=True)(coded_rnn_input)
    lstm2 = LSTM(128, unroll=True)(lstm1)
    #print(lstm1.get_shape())
    dense1 = Dense(encode_size, activation='relu')(lstm2)
    #print(dense1.get_shape())
    physics = Model(input=[coded_rnn_input], output=[dense1]);
   
    physics.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    '''

    def plottrain(past, present, future):
        fig, axes = plt.subplots(1, 7)
        for i, ax in enumerate(axes.flat):
            if i < 3:
                ax.imshow(past[i,0,:,:], cmap='gray')
            elif i == 3:
                ax.imshow(present[0,0,:,:], cmap='gray')
            else:
                print(future.shape)
                ax.imshow(future[i-4,0,:,:], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()




    train_or_load_lstm = False
    single_frame = True
    if not os.path.isdir("single_predictions"):
        os.mkdir('single_predictions')
    if not os.path.isdir("multi_predictions"):
        os.mkdir('multi_predictions')
    if train_or_load_lstm:
        nbEpoch = 100
        physics.load_weights("physics.h5")
        #encoder.trainable = False
        #decoder.trainable = False
        numtimes = len(train_list)-timeSize-5
        for epoch in range(1, nbEpoch + 1):
            indices = np.random.permutation(len(train_list));
            randperm = np.random.permutation(numtimes)
            for i in range(3, numtimes-1):
                t = randperm[i-2]

                train_batch = train_images_np[t:t+timeSize]
                train_label_batch = (train_images_np[t+timeSize+4])
                init_state = np.zeros((1,8,8,16))
                lstm_loss = physics.train_on_batch([init_state, np.reshape(train_batch[0], (1,128,128,1)),
                np.reshape(train_batch[1], (1,128,128,1)), 
                np.reshape(train_batch[2], (1,128,128,1)),
                np.reshape(train_batch[3], (1,128,128,1))], 
                #[np.reshape(train_label_batch[0], (1,8,8,16)),
                #np.reshape(train_label_batch[1], (1,8,8,16)),
                [np.reshape(train_label_batch, (1,128,128,1))])
                randInd = indices[i*batchSize:(i+1)*batchSize]
                #codec_loss = codec_only.train_on_batch(train_images_np[randInd], train_images_np[randInd])
            print(lstm_loss, "Epoch",  np.ceil(float(i)/numtimes+(epoch-1)))

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
        start = 35
        generate_codes = np.reshape(test_images_np[start:start+timeSize], (timeSize,128,128,1))
        print(generate_codes.shape)
        past = np.reshape(test_images_np[5:5+timeSize], (timeSize, im_size, im_size, 1))
        mse = 0
        init_state = np.zeros((1,8,8,16))
        iteration = len(test_list)-timeSize - start - 1
        for j in range(0, iteration - 5):

            #index = np.random.randint(3, len(test_list))
            index = j + start + 1
            if single_frame:
                generate_codes = np.reshape(test_images_np[index:index+timeSize], (timeSize,128,128,1))
            #print("test code shape", test_codes.shape)
            [physics_output] = physics.predict([init_state,np.reshape(generate_codes[0], (1,128,128,1)),
                np.reshape(generate_codes[1], (1,128,128,1)), 
                np.reshape(generate_codes[2], (1,128,128,1)),
                np.reshape(generate_codes[3], (1,128,128,1))])
            #init_state = hidden_state_physics
            #for t in range(timeSize-1):
            #    generate_codes[t,:,:,:] = generate_codes[t+1,:,:,:]
            #generate_codes[timeSize-1,:,:,:] = physics_output
            
            '''
            physics_output = physics.predict([init_state,np.reshape(generate_codes[0], (1,8,8,16)),
                np.reshape(generate_codes[1], (1,8,8,16)), 
                np.reshape(generate_codes[2], (1,8,8,16)),
                np.reshape(generate_codes[3], (1,8,8,16))])
            
            for t in range(timeSize-1):
                generate_codes[t,:,:,:] = generate_codes[t+1,:,:,:]
            generate_codes[timeSize-1,:,:,:] = physics_output[2]

            physics_output = physics.predict([init_state,np.reshape(generate_codes[0], (1,8,8,16)),
                np.reshape(generate_codes[1], (1,8,8,16)), 
                np.reshape(generate_codes[2], (1,8,8,16)),
                np.reshape(generate_codes[3], (1,8,8,16))])

            for t in range(timeSize-1):
                generate_codes[t,:,:,:] = generate_codes[t+1,:,:,:]
            generate_codes[timeSize-1,:,:,:] = physics_output[2]

            physics_output = physics.predict([init_state,np.reshape(generate_codes[0], (1,8,8,16)),
                np.reshape(generate_codes[1], (1,8,8,16)), 
                np.reshape(generate_codes[2], (1,8,8,16)),
                np.reshape(generate_codes[3], (1,8,8,16))])
            
            for t in range(timeSize-1):
                generate_codes[t,:,:,:] = generate_codes[t+1,:,:,:]
            generate_codes[timeSize-1,:,:,:] = physics_output[2]

            physics_output = physics.predict([init_state,np.reshape(generate_codes[0], (1,8,8,16)),
                np.reshape(generate_codes[1], (1,8,8,16)), 
                np.reshape(generate_codes[2], (1,8,8,16)),
                np.reshape(generate_codes[3], (1,8,8,16))])
            '''


            
            
            


            #Reshaper

            guessed = physics_output
            #guessed = decoder.predict(physics_output)
            for t in range(4-1):
            #    print(t)
                past[t,:,:,:] = past[t+1,:,:,:]
            past[3,:,:,:] = guessed
            #print(guessed.shape)
            #print(test_images_np[j+4,0,:,:].shape)
            #res = np.mean(np.square(train_images_np[300,0,:,:] - train_images_np[3+start,0,:,:]))
            #print("Res %f" % res)
            concat_guess = np.concatenate((np.squeeze(guessed), test_images_np[index+timeSize+4,:,:,0]), axis=1)
            mse = mse + np.mean(np.square(np.squeeze(guessed)- test_images_np[index+timeSize+4,:,:,0]))
            im = Image.fromarray((concat_guess*255).astype(np.uint8))
            if single_frame:   
                filename = 'single_predictions/future%d.png' % j
            else:
                filename = 'multi_predictions/future%d.png' % j 
            print(filename)
            im.save(filename)

        print(mse/iteration )
            #A = raw_input()
        #plottrain(test_images_np[index+2:index+timeSize], guessed, test_images_np[index+6:index+9])
        
