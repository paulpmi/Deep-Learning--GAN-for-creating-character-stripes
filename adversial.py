from discriminator import Discriminator
from generator import Generator


class Adversial:
    def __init__(self, path):
        l = Loader(path)
        self.input, self.output = l.load()
        self.model = Sequential()

    def createModel(self):
        g = self.Generator()
        d = self.Discriminator()
        self.model.add(g.createModel())
        self.model.add(d.createModel())
    
        return self.model.summary()

    def compileModel(self, lossFunction, optimizer):
        return self.model.compile(loss = lossFunction, optimizer = optimizer, metrics = ['accuracy'])

    def train(self):
        pass