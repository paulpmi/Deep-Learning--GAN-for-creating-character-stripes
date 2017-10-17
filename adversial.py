from discriminator import *
from generator import Generator
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

class Adversial:
    def __init__(self, path):
        l = Loader(path)
        self.input, self.output = l.load()
        self.model = Sequential()
        self.generator = Generator()
        self.generator.load(path)
        self.discriminator = Discriminator(path)

    def createModel(self, gnr, dnr):
        self.model.add(self.generator.createModel(gnr))
        self.model.add(self.discriminator.createModel(dnr))
    
        return self.model.summary()

    def compileModel(self, lossFunction, optimizer):
        return self.model.compile(loss = lossFunction, optimizer = optimizer, metrics = ['accuracy'])

    def train(self, x, y):
        return self.model.fit(x, y, epochs=2)


class DGAN:
    def __init__(self, path):
        l = Loader(path)
        self.input, self.output = l.load()
        #self.model = Sequential()
        self.generator = Generator()
        self.generator.load(path)
        self.generator.createModel(32)
        self.generator.compileModel('binary_crossentropy', 'adam')
        
        self.discriminator = Discriminator(path)
        self.discriminator.createModel(32)
        self.discriminator.compileModel('binary_crossentropy', 'adam')

        self.adversial = Adversial(path)
        self.adversial.createModel(32, 32)
        self.adversial.compileModel('binary_crossentropy', 'adam')

    def start(self, loop):
        for looper in range(loop):
            fakeImg = self.generator.train()
            print("FAKE")
            print(fakeImg.shape)
            print("REAL")
            print(self.input.shape)
            inputImg = np.vstack((self.input, fakeImg))
            o = []
            for i in range(fakeImg.shape[0]):
                o.append(0)
            output = np.asarray(o)

            print("FAKE OUTPUT")
            print(output.shape)
            print("OUTPUT")
            print(self.output.shape)

            outputImg = np.vstack((self.output, output))
            for i in range(4):
                outputImg = np.vstack((outputImg, output))

            print("INPUT")
            print(inputImg.shape)
            print("OUTPUT")
            print(outputImg.shape)

            discriminator_loss = self.discriminator.model.train_on_batch(inputImg, outputImg)
            
            fake2 = self.generator.train()
            faker = np.vstack((fake2, fakeImg))

            forcedOutput =  np.vstack((self.output, self.output))
            for i in range(4):
                forcedOutput = np.vstack((forcedOutput, self.output))

            adversial_loss = self.adversial.model.train_on_batch(faker, forcedOutput) # should be: fakeImg, self.output


        generated = self.generator.train()
        j = 0
        for i in generated:
            img = Image.fromarray(i, 'RGB')
            print(j)
            img.save('my'+str(j)+'.png')
            j+=1

algorithm = DGAN('./resizedData')
algorithm.start(5)