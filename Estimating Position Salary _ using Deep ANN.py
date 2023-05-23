# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)
y = y.reshape(len(y),1)
# print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
# print(X)
# print(y)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'Adam', loss = 'MeanSquaredError' , metrics = ['mae'])

# Training the ANN on the Training set
ann.load_weights('Weights.h1')
# ann.fit(X, y, batch_size = 2, epochs = 1000)
# ann.save_weights('Weights.h1')

# Part 4 - Making the predictions and evaluating the model

# Predicting a new result
Level= 6.5
print('\n Estimated Position Salary :', sc_y.inverse_transform(ann.predict(sc_X.transform([[Level]])).reshape(-1, 1))[0][0])

# Visualising results
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.scatter([Level], sc_y.inverse_transform(ann.predict(sc_X.transform([[Level]])).reshape(-1, 1)), color='green', s=200, marker='*') # this has been added to show the single prediction result
plt.plot(X_grid, sc_y.inverse_transform(ann.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color = 'blue')
plt.title('Estimated Salary (Deep ANN)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
