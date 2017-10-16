import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import os

class Loader:
    def __init__(self, path):
        self.path = path
        self.imgArray = None
    
    def load(self):
        i = 0
        for filename in os.listdir(self.path):
            image_data = imread(self.path + '/' + filename).astype(np.float32)
            if i == 0:
                self.imgArray = image_data
                i+=1
            else:
                np.vstack((self.imgArray, image_data))
        return self.imgArray

