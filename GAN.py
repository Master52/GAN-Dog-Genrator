
from keras.models import Sequential
from keras.layers import Conv2D,UpSampling2D,Reshape,Activation,Dense,BatchNormalization,LeakyReLU
from keras.optimizers import Adam
import numpy as np
import cv2 as cv

class GAN:
    def __init__(self,generator,discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.model = self.__initializeGAN()

    def __initializeGAN(self):
        self.discriminator.trainable = False
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        opt = Adam(lr = 0.0002,beta_1 = 0.5)
        model.compile(loss = 'binary_crossentropy',optimizer = opt)
        return model

    def summary(self):
        print("MODEL SUMMRY")
        print("GENERATOR")
        print(self.generator.summary())
        print("DISCRIMINATOR")
        print(self.discriminator.summary())
        print("OVER ALL")
        print(self.model.summary())

    def __load_fake_img(self,seed):
        noise = np.random.normal(0,1,size = 32 * seed).reshape(32,seed)
        X = self.generator.predict(noise)
        y = np.zeros((32,1))
        return X,y

    def __load_real_img(self,img):
        ix = np.random.randint(0,img.shape[0],32)
        img = img[ix]
        y = np.ones((img.shape[0],1))
        return img,y

    def run(self,images,seed = 100,epoches = 50):
        for i in range(epoches):
            self.discriminator.trainable = True
            real_img,real_label = self.__load_real_img(images)
            real_loss = self.discriminator.train_on_batch(real_img,real_label)
            fake_img,fake_label = self.__load_fake_img(seed)
            fake_loss = self.discriminator.train_on_batch(fake_img,fake_label)
            noise = np.random.normal(0,1,size = 32 * seed).reshape(32,seed)
            self.discriminator.trainable = False
            g_loss = self.model.train_on_batch(noise,np.ones((32,1)))
            img = self.generator.predict(np.random.normal(0,1,size = seed).reshape(1,seed))[0]
            img = ((img + 1)*127.5).astype(np.uint8)
            cv.imwrite('img.png',img)
            print(f'epoch:{i} \nreal_loss: {real_loss}\nfake_loss:{fake_loss}\ng_loss:{g_loss}\n')

        img = self.generator.predict(np.random.normal(0,1,size = seed).reshape(1,seed))[0]
        img = ((img + 1)*127.5).astype(np.uint8)
        return img
