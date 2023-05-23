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

# Training XGBoost on the whole dataset
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X, y)

# Predicting a new result
Level= 6.5
print('\n Estimated Position Salary :', regressor.predict([[Level]]).reshape(-1, 1)[0][0])

# Visualising the results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.scatter([Level], regressor.predict([[Level]]).reshape(-1, 1), color='green', s=200, marker='*') # this has been added to show the single prediction result
plt.plot(X_grid, regressor.predict(X_grid).reshape(-1, 1), color = 'blue')
plt.title('Estimated Salary (XGBoost)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
