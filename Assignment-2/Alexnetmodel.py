import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
np.random.seed(1000)


def alexnetmodel():

	#defining the Alexnet model
	model = Sequential()

	# 1st Convolutional Layer
	model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
	strides=(4,4), padding='valid'))
	model.add(Activation('relu'))
	# Pooling 
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	# Batch Normalisation before passing it to the next layer
	model.add(BatchNormalization())

	# 2nd Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	# Batch Normalisation
	model.add(BatchNormalization())

	# 3rd Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Batch Normalisation
	model.add(BatchNormalization())

	# 4th Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Batch Normalisation
	model.add(BatchNormalization())

	# 5th Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	# Batch Normalisation
	model.add(BatchNormalization())

	# Passing it to a Dense layer
	model.add(Flatten())
	# 1st Dense layer
	model.add(Dense(4096, input_shape=(224*224*3,)))
	model.add(Activation('relu'))
	# To prevent overfit using dropout.
	model.add(Dropout(0.4))
	# Batch-Normalisation
	model.add(BatchNormalization())

	# 2nd Dense layer
	model.add(Dense(4096))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))
	# Batch-Normalisation
	model.add(BatchNormalization())

	# 3rd Dense layer
	model.add(Dense(1000))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))
	# Batch-Normalisation
	model.add(BatchNormalization())

	# Output Layer
	model.add(Dense(102))
	model.add(Activation('softmax'))
	
	# compile the model to make a use of categorical cross-entropy loss function and adadelta optimizer to get high accuracy.
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	return model
