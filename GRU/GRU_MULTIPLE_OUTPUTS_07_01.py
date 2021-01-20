'''
GRU for Watershed:
    -Lags as inputs;
    -t+n as outputs;
    -Forecasting results

Optimizaing:
    -Training_set: 11 years
    -Learning rate: 0.001
    -Optimizer: Adam
    -Neurons: 15 x4
    -Dropout: 0.5

Plan:   
    - Lags try: 1, 3, 5, 10, 30
    - Out try: 1
    - Selected Lag: 10
    - Outs: 1, 2, 3, 5 e 10
'''

#Imports
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas import DataFrame, concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import concatenate
import numpy as np
import os
from math import sqrt
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow import keras
import time

# Parametros alteraveis
path = "C:/Users/Vini/Desktop/Cloud_Google/Artigo Bacias/Final_Codes/GRU"

#Specifying the lag
n_days = 10 #Lag + n
n_features = 4 
n_out = 10 #Prediction t + n

save_path_name = 'LAG' + str(n_days) + 'OUT' + str(n_out)

full_name = os.path.join(path,save_path_name) 

if os.path.exists(full_name) is False:
  os.mkdir(full_name) # cria o diretorio se ele ja nao existe
else:
  print("Directory already exists, the files will be rewrite")


#Loading data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

#Loading Dataset
dataset = read_csv('database.csv')
values = dataset.values

#Manually specifing column names
dataset.columns = ['r_gaurama','r_mr','r_tapejara','q_mr']

#Marking all NA values with 0
dataset['r_gaurama'].fillna(0, inplace = True)
dataset['r_mr'].fillna(0, inplace = True)
dataset['r_tapejara'].fillna(0, inplace = True)
dataset['q_mr'].fillna(0, inplace = True)

#Specifing columns to plot
groups = [0,1,2,3]
i = 1

#Convert series to supervised learning
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True): #n_in = number of inputs; n_out = number of outputs
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	#input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	#forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#Setting all dataset as float
values = values.astype('float32')

#Normalizing features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

n_obs = n_days * n_features

#Frame as supervised learning
reframed = series_to_supervised(scaled, n_days, n_out)

#Dropping cols for t+n days as outputs
drop_cols = np.arange(n_obs,reframed.shape[1])

#Loop for as index parameter
z = -1
index = []
for i in range(len(drop_cols)):
  if z + n_features == i: # in this case it's 4 features
    index.append(i)
    z = z + n_features

drop_cols = np.delete(drop_cols, index)

reframed.drop(reframed.columns[[drop_cols]], axis=1, inplace=True) # columns that won't be predicted

'''
Defining and fitting the model
Spliting into train and test sets
Total of 12 years; *Set: 9 for training and 3 for testing
About 70% of data for training and 30$ for testing
'''

#Splitting
values = reframed.values
n_train_days = 365 * 11
train = values[:n_train_days, :]
test = values[n_train_days:, :]

#Splitting into input and outputs // CHANGE HERE // concatanete within -3
X_train, y_train = train[:, :n_obs], train[:, -n_out:]
X_test, y_test = test[:, :n_obs], test[:, -n_out:]

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], n_days, n_features)) #linhas train_X (exemplos); 1 coluna; colunas train_X (numero de parametros/variaveis)
X_test = X_test.reshape((X_test.shape[0], n_days, n_features))

'''
Defining the Model
'''
#Callbacks
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, 
                   patience = 30) 
csv_logger = CSVLogger(full_name + '/' + save_path_name + '_logger.csv', 
                       separator = ',', append = False)
model_checkpoint = ModelCheckpoint(full_name + '/' + save_path_name + '_GRU_train.hdf5', 
                                   monitor='val_loss',verbose = 1, save_best_only = True)

'''
Defining the Model
'''
#Network structure

#Defining learning rate
opt = keras.optimizers.Adam(learning_rate = 0.001)

model = Sequential()

model.add(GRU(50, input_shape = (X_train.shape[1], X_train.shape[2]),
               return_sequences = True))
model.add(Dropout(0.5))

model.add(GRU(50, return_sequences = True))
model.add(Dropout(0.5))

model.add(GRU(50, return_sequences = True))
model.add(Dropout(0.5))

model.add(GRU(50))
model.add(Dropout(0.5))

model.add(Dense(n_out, activation = 'linear'))

model.compile(loss = 'mse', optimizer = opt, metrics = ['mse'])

#Initial time
start_time = time.time()

#Fitting Network
history = model.fit(X_train, y_train, epochs = 300, batch_size = 73,
                    validation_data = (X_test, y_test), verbose = 2, 
                    shuffle = False, callbacks = [model_checkpoint,csv_logger,es])

#Final time
end_time = (time.time() - start_time)

with open(full_name + '/' + save_path_name + 'T' + str(i) + '_timing.txt', 'w') as out:
    out.write("The model timing is: %.2f seconds" % end_time)
    out.close()

'''
Evaluating the metrics of the Model
'''

#Making a prediction

y_pred = model.predict(X_test)

X_test = X_test.reshape((X_test.shape[0], n_days*n_features))

for i in range(y_train.shape[1]):
  #Inverting scaling for forecast 
  inv_y_pred = concatenate((np.expand_dims(y_pred[:,i],axis=1), X_test[:, :3]),axis=1)  # colocar laço for pra gerar pra cada saída? 
  inv_y_pred = scaler.inverse_transform(inv_y_pred)
  inv_y_pred = inv_y_pred[:,0] 

  np.savetxt(full_name + '/' + save_path_name + '_O' + str(i) + '_predicted.txt', inv_y_pred)

    
  #Inverting scale to actual/real values
  y_test_2 = y_test[:,i].reshape((len(y_test[:,i]), 1))
  inv_y = concatenate((np.expand_dims(y_test[:,i],axis=1), X_test[:, :3]),axis=1) # verificar o índice :3
  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:,0]

  np.savetxt(full_name + '/' + save_path_name + '_real.txt', inv_y)

  #Calculating metrics
  print('Metrics for t +',i)
  rmse = sqrt(mean_squared_error(inv_y, inv_y_pred))
  print('Test RMSE: %.3f' % rmse)
  mae = mean_absolute_error(inv_y, inv_y_pred)
  print('Test MAE: %.3f' % mae)
  print('_____________________')

  with open(full_name + '/' + save_path_name + 'O' + str(i) + '_metrics.txt', 'w') as out:
    out.write('Test RMSE: %.3f' % rmse + '\n')
    out.write('Test MAE: %.3f' % mae + '\n')
    out.close()

  #Plotting history
  plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss t + %i' %i)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend(['train', 'validation'], loc='upper right')
  plt.savefig(full_name + '/' + save_path_name + '_' + str(i) + '_loss.tif' , dpi = 300)

  #Plotting the predictions
  plt.figure()
  plt.plot(inv_y, label = 'Observed')
  plt.plot(inv_y_pred, label = 'Predicted')
  plt.title('Predicted vs Observed t + %i' %i)
  plt.xlabel('Days')
  plt.ylabel('q (m³/s)')
  plt.legend()
  plt.savefig(full_name + '/' + save_path_name + '_' + str(i) + '_predicted.tif' , dpi = 300)
