#here is the code that uses scene15 dataset to train the Alexnet model using deep convolution network.
#starts from impporting all the necessary library. 
import numpy as np
import pandas as pd
from keras import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import glob
from skimage import io
import os
import scipy.misc
from scipy.misc import imread, imresize
from keras import regularizers
import Alexnetmodel
import csv
import warnings
warnings.filterwarnings("ignore")

# Add the path of the dataset
datasets_path = 'S:\DL\Assign-2\scene' 

#loading the images and setting the size of the images.
def load_imgs(path,n=0):
    X=[] 
    Y=[]
    i=-1
    lbls = [] #array of the labels
    for lbl in os.listdir(path):
        back_path = os.path.join(path,lbl)
        lbls.append(lbl)
        i = i+1
        for filename in os.listdir(back_path):
            image_path = os.path.join(back_path,filename)
            img = image.load_img(image_path,target_size=(224,224))
            img = image.img_to_array(img)
            img[:,:,0] -= 123.68
            img[:,:,1] -= 116.78
            img[:,:,2] -= 103.94
            Y.append(i)
            X.append(img)
    return X,Y,lbls
#Loading the images from desired path
x_train,y,label_data = load_imgs(datasets_path)
X = np.array(x_train)
Y = np.array(y)
print(X.shape)

#divide the datasete in train(80%) and test(20%) part
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
number_of_classes = 15
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
print(Y_test.shape)

#Alexnet model for classification
model = Alexnetmodel.alexnetmodel()
model.summary()
	
#train the data by fitting it in model
Final = model.fit(X_train, Y_train,batch_size=128,epochs=10)

#model evaluation
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
