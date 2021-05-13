#here is the code that uses caltech101 dataset to train the alexnet model using deep convolution network.
#starts from impporting all the necessary library.
import random
import os
import numpy as np
import keras
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
import Alexnetmodel
from tensorflow.keras.utils import to_categorical


# Loading dataset and distributing in categories
root = 'Caltech101'
exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplanes', 'Faces_easy', 'Faces']
train_splt, val_splt = 0.7, 0.15 #dataset division for training(70%) testing(15%) and validation(15%)

Categories = [x[0] for x in os.walk(root) if x[0]][1:]
Categories = [c for c in Categories if c not in [os.path.join(root, e) for e in exclude]]
print(Categories)

#getting the image from desire path
def get_img(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

data = []
for c, Category in enumerate(Categories):
    images = [os.path.join(dp, f) for dp, dn, filenames 
              in os.walk(Category) for f in filenames 
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    for img_path in images:
        img, x = get_img(img_path)
        data.append({'x':np.array(x[0]), 'y':c})

# counting the number of classes
num_classes = len(Categories)

# randomly shuffle the input data images
random.shuffle(data)

#split the dataset into train test and validation part and convert it in array form
idx_val = int(train_splt * len(data))
idx_test = int((train_splt + val_splt) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]
print(y_test)

# normalize data
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert labels to one-hot vectors
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)
print(y_test.shape)

print("finished loading %d images from %d Categories"%(len(data), num_classes))
print("train / validation / test split: %d, %d, %d"%(len(x_train), len(x_val), len(x_test)))
print("training data shape: ", x_train.shape)
print("training labels shape: ", y_train.shape)

#Calling Alexnet model
model = Alexnetmodel.alexnetmodel()
model.summary()

# compile the model to make a use of categorical cross-entropy loss function and adadelta optimizer to get high accuracy.
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#train the data by fitting it in model.
Final = model.fit(x_train, y_train,batch_size=128,epochs=100,validation_data=(x_val, y_val))

#model evaluation
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
