from discriminator import *
from generator import Generator
from upscaler import *
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import PIL

class Adversial:
    def __init__(self, path, g, d):
        l = Loader(path)
        self.input, self.output = l.load()
        self.model = Sequential()
        self.generator = g
        #self.generator.load(path)
        self.discriminator = d

    def createModel(self, gnr, dnr):
        self.model.add(self.generator.getModel())
        self.model.add(self.discriminator.getModel())
    
        return self.model.summary()

    def compileModel(self, lossFunction, optimizer):
        return self.model.compile(loss = lossFunction, optimizer = optimizer, metrics = ['accuracy'])

    def train(self, x, y):
        return self.model.fit(x, y, epochs=2)

    def generate(self):
        return self.generator.train()


class DGAN:
    def __init__(self, path):
        l = Loader(path)
        self.input, self.output = l.load()
        #self.model = Sequential()
        self.generator = Generator()
        self.generator.load(path)
        self.generator.createModel(16)
        self.generator.compileModel('binary_crossentropy', 'adam')
        
        self.discriminator = Discriminator(path)
        self.discriminator.createModel(32)
        self.discriminator.compileModel('binary_crossentropy', 'adam')

        self.adversial = Adversial(path, self.generator, self.discriminator)
        self.adversial.createModel(16, 32)
        self.adversial.compileModel('binary_crossentropy', 'adam')

    def start(self, loop):
        for looper in range(loop):
            fakeImg = self.adversial.generator.train()
            print("FAKE")
            print(fakeImg.shape)
            print("REAL")
            print(self.input.shape)
            inputImg = np.vstack((self.input, fakeImg))
            """
            for i in self.input:
                print("POWERPLANT")
                print(i.shape)
                inputImg = np.vstack((i, fakeImg))
            """
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

            for k in outputImg:
                print("KOALAPLANT")
                print(k.shape)
                k = np.repeat(k,2)
                print(k.shape)
                discriminator_loss = self.adversial.discriminator.model.train_on_batch(inputImg, k)
            
            fake2 = self.adversial.generator.train()
            faker = np.vstack((fake2, fakeImg))

            forcedOutput =  np.vstack((self.output, self.output))
            for i in range(4):
                forcedOutput = np.vstack((forcedOutput, self.output))

            power = []
            for n in range(4):
                a = numpy.random.rand(28,28,3) * 28
                #print(a)
                #im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
                power.append(a)

            power = np.asarray(power)           
            for k in forcedOutput:
                k = np.repeat(k,2)
                adversial_loss = self.adversial.model.train_on_batch(inputImg, k) # should be: fakeImg, self.output


        generated = self.adversial.generate()
        #uS = Upscaler(generated)
        #generated = uS.scale()
        j = 0
        for i in generated:
            img = Image.fromarray(i, 'RGB')
            img.resize((1280, 1024), PIL.Image.ANTIALIAS)
            print(j)
            img.save('my'+str(j)+'.png')
            j+=1

algorithm = DGAN('./resizedData')
algorithm.start(2)