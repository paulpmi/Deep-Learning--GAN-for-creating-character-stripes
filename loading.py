import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import os

class Loader:
    def __init__(self, path):
        self.path = path
        self.imgArray = None
        self.output = []
    
    def load(self):
        """
        i = 0
        for filename in os.listdir(self.path):
            image_data = imread(self.path + '/' + filename).astype(np.float32)
            if i == 0:
                self.imgArray = image_data
                self.output.append(1)
                i+=1
            else:
                np.vstack((self.imgArray, image_data))
                self.output.append(1)
        print(self.imgArray.shape)
        return self.imgArray, self.output
        """
        l = []
        o = []
        for filename in os.listdir(self.path):
            l.append(imread(self.path + '/' + filename))
            o.append(1)
        self.imgArray = np.asarray(l)
        self.output = np.asarray(o)
        return self.imgArray, self.output