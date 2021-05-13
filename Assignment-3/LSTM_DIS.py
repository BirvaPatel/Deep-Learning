import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time

input_file="DIS.csv"

# Transfering to dataset matrics
def create_dataset(ds, timestep=1):
	x_Data, y_Data = [], []
	for i in range(len(ds)-timestep-1):
		string = ds[i:(i+timestep), 0]
		x_Data.append(string)
		y_Data.append(ds[i + timestep, 0])
	return np.array(x_Data), np.array(y_Data)


np.random.seed(5)
df = read_csv(input_file, header=None, index_col=None, delimiter=',')
Y = df[5].values
ds=Y.reshape(-1, 1)

# Dataset Normalization
norm = MinMaxScaler(feature_range=(0, 1))
ds = norm.fit_transform(ds)

# Spliting dataset into Train(50%) and Test(50%)
Tr_size = int(len(ds) * 0.5)
Te_size = len(ds) - Tr_size
Tr, Te = ds[0:Tr_size,:], ds[Tr_size:len(ds),:]

# Reshape Train and Test data with timestep
timestep = 240
x_Train, y_Train = create_dataset(Tr, timestep)
x_Test, y_Test = create_dataset(Te, timestep)
x_Train = np.reshape(x_Train, (x_Train.shape[0], 1, x_Train.shape[1]))
x_Test = np.reshape(x_Test, (x_Test.shape[0], 1, x_Test.shape[1]))

# Define LSTM network
model = Sequential()
model.add(LSTM(25, input_shape=(1, timestep)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(x_Train, y_Train, epochs=1000, batch_size=256, verbose=1)

# Prediction Assumption
Predict_Tr = model.predict(x_Train)
Predict_Te = model.predict(x_Test)

# Transform Prediction
Predict_Tr = norm.inverse_transform(Predict_Tr)
y_Train = norm.inverse_transform([y_Train])
Predict_Te = norm.inverse_transform(Predict_Te)
y_Test = norm.inverse_transform([y_Test])

# Calculation of RMSE(root mean square error)
Score_Tr = math.sqrt(mean_squared_error(y_Train[0], Predict_Tr[:,0]))
print('Tr Score: %.2f RMSE' % (Score_Tr))
Score_Te = math.sqrt(mean_squared_error(y_Test[0], Predict_Te[:,0]))
print('Te Score: %.2f RMSE' % (Score_Te))

# For plotting,shif train prediction 
Tr_Predictplot = np.empty_like(ds)
Tr_Predictplot[:, :] = np.nan
Tr_Predictplot[timestep:len(Predict_Tr)+timestep, :] = Predict_Tr

# For plotting,shift test prediction 
Te_Predictplot = np.empty_like(ds)
Te_Predictplot[:, :] = np.nan
Te_Predictplot[len(Predict_Tr)+(timestep*2)+1:len(ds)-1, :] = Predict_Te

# Plotting Baseline
plt.plot(norm.inverse_transform(ds))
plt.plot(Tr_Predictplot,color='red')
Te_Prices=norm.inverse_transform(ds[Te_size+timestep:])
print('TestPredictions:')
print(Predict_Te)    

# Storing prediction
df = pd.DataFrame(data={"prediction": np.around(list(Predict_Te.reshape(-1)), decimals=2), "test_price": np.around(list(Te_Prices.reshape(-1)), decimals=2)})
df.to_csv("lstm_result.csv", sep=';', index=None)

# plot the Prediction and actualprice 
plt.plot(Te_Predictplot,color='yellow')
plt.legend(['Actual Price','train', 'test'], loc='upper left')
plt.show()


