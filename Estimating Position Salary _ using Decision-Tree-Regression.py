# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
Level= 6.5
print('\n Estimated Posotion Salary :', regressor.predict([[Level]])[0])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.scatter([Level], regressor.predict([[Level]]), color='green', s=200, marker='*') # this has been added to show the single prediction result
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Estimated Salary (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()