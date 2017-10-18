import PIL
from PIL import Image

class Resizer:
    def __init__(self, size):
        self.size = size
        self.image = None

    def resize(self):
        i = 0
        for filename in os.listdir(self.path):
            img = Image.open(self.path + '/' + filename).convert('RGB')
            hpercent = (baseheight / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(hpercent)))
            img.save('./resizedData/' +' resized_image'+str(i))
            image_data = imread(self.path + '/' + filename).astype(np.float32)
            i+=1



baseheight = 224
for i in range(1, 4):
    img = Image.open('./data/k.' + str(i) + '.jpg').convert('RGB')
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((28, 28), PIL.Image.ANTIALIAS)
    img.save('./train/resized_image' + str(i) + '.jpg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
image_data = imread('./train/resized_image1.jpg').astype(np.float32)
print ('Size: ', image_data.size)
print ('Shape: ', image_data.shape)
