import numpy as np
import keras
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit, KFold, cross_val_score
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class ELM (BaseEstimator, ClassifierMixin):

	#initialize the variables 
	def __init__(self,hid_num,a=1):
		self.hid_num = hid_num
		self.a = a
		
	# Appling the sigmoid
	def Sigmoid(self, x):
		return 1 / (1 + np.exp(-self.a * x))

	# Adding bias between layers
	def Bias(self, X):
		return np.c_[X, np.ones(X.shape[0])]

	def Ltov(self, n, label):
		return [-1 if i != label else 1 for i in range(1, n + 1)]

	# fit all the data to the model
	def fit(self, X, y):
		self.out_num = max(y)
		if self.out_num != 1:
			y = np.array([self.Ltov(self.out_num, _y) for _y in y])
		X = self.Bias(X)
		np.random.seed()
		self.W = np.random.uniform(-1., 1.,(self.hid_num, X.shape[1]))# weights between i/p layer and hid layer
		invrs_H = np.linalg.pinv(self.Sigmoid(np.dot(self.W, X.T))) # find inverse weight matrix
		self.beta = np.dot(invrs_H.T, y)
		return self

	# prediction using the sigmoid and inverse function
	def predict(self, X):
		
		invrs_H = self.Sigmoid(np.dot(self.W, self.Bias(X).T))
		y = np.dot(invrs_H.T, self.beta)
		if self.out_num == 1:
			return np.sign(y)
		else:
			return np.argmax(y, 1) + np.ones(y.shape[0])

#Load the IRIS dataset 			
iris = load_iris()
X = iris.data  
y = iris.target

#data spliting
x_train, x_test , y_train, y_test = train_test_split(X,y, test_size= 0.3)
num_classes = 3
#Normalization using to_categorical function
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
hid_num = 100

#Using ELM classifier
e = ELM(hid_num)

#K-Fold cross validation
ave = 0
for i in range(10):
    cv = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(e, X, y,cv=cv, scoring='accuracy')
    ave += scores.mean()

ave /= 10

#Printing the final accuracy
print("Accuracy: %0.3f " % (ave*100))
