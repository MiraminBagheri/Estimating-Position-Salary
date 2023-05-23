# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y.ravel())

# Predicting a new result
Level= 6.5
print('\n Estimated Position Salary :', sc_y.inverse_transform(regressor.predict(sc_X.transform([[Level]])).reshape(-1, 1))[0][0])

# Visualising the results
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.scatter([Level], sc_y.inverse_transform(regressor.predict(sc_X.transform([[Level]])).reshape(-1, 1)), color='green', s=200, marker='*') # this has been added to show the single prediction result
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color = 'blue')
plt.title('Estimated Salary (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
