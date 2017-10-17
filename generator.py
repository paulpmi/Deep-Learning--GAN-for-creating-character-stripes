from keras.layers.core import Dense, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Deconv2D, ZeroPadding2D, Input, UpSampling2D, Flatten, Activation
from keras.models import Sequential

from loading import Loader

import numpy

class Generator:
    def __init__(self):
        self.model = Sequential()
        self.input = None
        self.output = None

    def load(self, path):
        l = Loader(path)
        self.input, self.output = l.load()
        print(self.input.shape)

    def createModel(self, imgDim):
        self.model.add(Conv2D(32, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))
        #self.model.add(Flatten())
        #self.model.add(Dense(imgDim**2*3))
        #self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(Activation('relu'))
        #self.model.add(Reshape((imgDim, imgDim, 3)))
        self.model.add(Deconv2D(16, 5, padding='same'))
        self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(Activation('relu'))
        self.model.add(Deconv2D(8, 5, padding='same', activation='relu'))
        self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(Deconv2D(1, 5, padding='same'))
        self.model.add(Activation('sigmoid'))
        #self.model.add(Flatten())
        #self.model.add(Dense(1))

        return self.model.summary()

    def compileModel(self, lossFunction, optimizer):
        return self.model.compile(loss = lossFunction, optimizer = optimizer, metrics = ['accuracy'])


    def train(self):
        return self.model.predict(self.input)

g = Generator()
g.load('./resizedData')
Z = numpy.random.rand(224,224,3)
g.createModel(7)
print(g.compileModel('binary_crossentropy', 'adam'))
print(g.train())

#from pylab import imshow, show, get_cmap
#from numpy import random
#Z = numpy.random.uniform(-1.0, 1.0, size=[256, 100])   # Test data
#imshow(Z, cmap=get_cmap("Spectral"), interpolation='nearest')
#show()