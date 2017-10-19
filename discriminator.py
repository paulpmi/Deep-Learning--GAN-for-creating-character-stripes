from keras.layers.core import Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Deconv2D, ZeroPadding2D, Input, UpSampling2D, Flatten
from keras.models import Sequential

import numpy

from loading import Loader

class Discriminator:
    def __init__(self, path):
        l = Loader(path)
        self.input, self.output = l.load()
        print(self.input.shape)
        self.model = Sequential()

    def createModel(self, size):
        self.model.add(Conv2D(size, (3, 3), input_shape = (100, 100, 3), activation = 'relu'))
        self.model.add(Conv2D(size, (3, 3), activation = 'relu'))
        self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(size*2, (3, 3), activation = 'relu'))
        self.model.add(Conv2D(size*2, (3, 3), activation = 'relu'))
        self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))

        return self.model

    def compileModel(self, lossFunction, optimizer):
        return self.model.compile(loss = lossFunction, optimizer = optimizer, metrics = ['accuracy'])

    def getModel(self):
        return self.model

    def train(self):
        return self.model.fit(self.input, self.output, epochs=1)

"""
d = Discriminator('./resizedData')
print(d.createModel(32))
print(d.compileModel('binary_crossentropy', 'adam'))
print(d.train())
"""