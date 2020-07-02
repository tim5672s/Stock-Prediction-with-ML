import numpy as np
import math 
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import date
plt.style.use('fivethirtyeight')

today = str(date.today())


# Get Stock Informatio
df = web.DataReader('0P0000M5E1.F', data_source='yahoo', start='2012-01-01', end=today)

#  Get nr of rows and cols in data set
df.shape

# Visualize closing prize
plt.figure(figsize=(8,4))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=10)
plt.ylabel('Close price USD')



# Create new Data frome with close-Clo
data = df.filter(['Close'])
#Convert df to numpy array
dataset = data.values
#Get the number of rows to train the model
training_data_len = math.ceil(len(dataset) * .8)

# Scale data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
# Create scaled training data set

training_data = scaled_data[0:training_data_len, :]
# Split data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()


# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape data
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))

# Build LTSM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile MOdel
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=20)

# Create testing data set
# Create new array containing scaled values
test_data = scaled_data[training_data_len-60:, :]
#Create data set x_test, y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


# Convert data to numpy array
x_test = np.array(x_test)
# Reshape data from 2D to 3D
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get Model predicted price vals
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Eval Model
# Get root mean sqare (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2 )

#  Plot
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#vis data
plt.figure(figsize=(8,4))
plt.title('Model')
plt.xlabel('Data')
plt.ylabel('Close Price USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


#Show actual price and predict. price
print(valid)

# Get the quote 
quote = web.DataReader('0P0000M5E1.F', data_source='yahoo', start='2012-01-01', end=today)
#Create new dataframe
new_df = quote.filter(['Close'])

#Get last 60 days closing price and convert the dataframe to an array 
last_60days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60days_scaled = scaler.transform(last_60days)
X_test = []
#Append the past 60 days
X_test.append(last_60days_scaled)
#Convert X_test dataset to np array
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

# Get the quote 
quote2 = web.DataReader('0P0000M5E1.F', data_source='yahoo', start='2019-12-18', end=today)
print(quote2['Close'])