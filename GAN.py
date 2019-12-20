
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
MAX_IMG = 252

class GAN:
    def __init__(self,generator,discriminator):

        self.generator = generator
        self.discriminator = discriminator

        self.model = self.__initializeGAN()

    def __initializeGAN(self):
        self.discriminator.trainable = False # So we dont train discriminator while we train genarator
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

    def __get_noise(self,seed,size = 1):
            return np.random.normal(0,1,size = size * seed).reshape(size,seed)

    def __load_fake_img(self,seed):
        noise = self.__get_noise(seed,size = MAX_IMG)

        X = self.generator.predict(noise)
        y = np.zeros((MAX_IMG,1))
        return X,y

    def __load_real_img(self,img):
        ix = np.random.randint(0,img.shape[0],MAX_IMG)
        img = img[ix]
        y = np.ones((img.shape[0],1))
        return img,y

    def __convert(self,img):
            return ((img + 1)*127.5).astype(np.uint8)

    def run(self,images,seed = 100,batches = 50):
        for i in range(batches):
            #Trainning Discriminator
            self.discriminator.trainable = True

            real_img,real_label = self.__load_real_img(images)
            real_loss = self.discriminator.train_on_batch(real_img,real_label)
            fake_img,fake_label = self.__load_fake_img(seed)
            fake_loss = self.discriminator.train_on_batch(fake_img,fake_label)

            #Trainning Generator
            self.discriminator.trainable = False

            noise = self.__get_noise(size = MAX_IMG,seed = seed)
            g_loss = self.model.train_on_batch(noise,np.ones((MAX_IMG,1)))

            print(f'real_loss:{real_loss} \n fake_loss:{fake_loss} \n g_loss:{g_loss}')

        img = self.generator.predict(self.__get_noise(seed = seed))[0]
        img = self.__convert(img)
        return img
