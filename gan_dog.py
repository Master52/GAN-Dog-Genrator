#! /bin/python3 

import cv2 as cv
import numpy as np

import os
from keras.models import Sequential
from keras.layers import Conv2D,UpSampling2D,Reshape,Activation,Dense,BatchNormalization,LeakyReLU,Flatten,Dropout
import tensorflow as tf
from keras.optimizers import Adam
from GAN import GAN


PATH = "dataset/Images"
IMG_SIZE = tuple((64,64))
MAX_IMG= 32
SEED = 100

IMAGES =  np.asarray([
                  cv.resize(
                  cv.imread(os.path.join(PATH,f)),
                  IMG_SIZE,interpolation = cv.INTER_AREA) 
                  for f in os.listdir(PATH)[0:MAX_IMG]
                  ])

def define_generator(channel,latent_space = SEED):
    model = Sequential()
    model.add(Dense(4*4*256,activation = 'relu' ,input_dim = latent_space))
    model.add(Reshape((4,4,256)))
    #upsampling image to 8 * 8
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size= 3,padding = 'same'))
    model.add(Activation("relu"))
    #upsampling image to 16 * 16
    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size = 3,padding = 'same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    #upsampling image to 32 * 32
    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size = 3,padding = 'same'))
    model.add(Activation("relu"))

    #OUPUT SAMPLEING 64*64
    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size = 3,padding = 'same'))
    model.add(Activation("relu"))

    #Final CNN Layer
    model.add(Conv2D(channel,kernel_size = 3,padding ='same'))
    model.add(Activation('tanh'))

    return model


def define_descriminator(image_shape):
    model = Sequential()
    model.add(Conv2D(32,kernel_size = 3,strides = 2 , input_shape = image_shape,padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64,kernel_size = 3,strides = 2,padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))

    #Add drop out
    model.add(Dropout(0.25))
    model.add(Conv2D(128,kernel_size = 3,strides = 2,padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))

    #Add drop out
    model.add(Dropout(0.25))
    model.add(Conv2D(256,kernel_size = 3,strides = 2,padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))

    #Add drop out
    model.add(Dropout(0.25))
    model.add(Conv2D(512,kernel_size = 3,strides = 2,padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))

    #Add drop out
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1,activation = 'sigmoid'))

    opt = Adam(lr=0.0002,beta_1 = 0.5)
    model.compile(loss = 'binary_crossentropy',optimizer = opt,metrics = ['accuracy'])

    return model


def load_real_img(IMAGES):
    IMAGES = np.asarray(IMAGES,np.float32)
    IMAGES = IMAGES / 127.5 -1
    return IMAGES


if __name__ == '__main__':
    generator = define_generator(channel = 3)
    discriminator = define_descriminator(IMAGES[0].shape)
    gan = GAN(generator,discriminator)
    for i in range(200):
        print(f'batche:{i}')
        img = gan.run(load_real_img(IMAGES),SEED)
        e = str(i) + ".png"
        cv.imwrite(e,img)
        print(f'Done:{i}')

#TODO 
# * D-dimensional noise vecot (64,64,3)[1000] -> G(64,64,3)[1000] Genrator  network -> Fake Image  DONE

#
# [Fake Image ->  Discrimator network <- Real Image]->sample image
# sample image -> fine tunning -> genrator 



