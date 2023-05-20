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

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Predicting a new result
Level= 6.5
# Predicting a new result with Linear Regression
print('\n Estimated Position Salary (Linear Regression):', sc_y.inverse_transform(lin_reg.predict(sc_X.transform([[Level]])).reshape(-1, 1))[0][0])
# Predicting a new result with Polynomial Regression
print('\n Estimated Position Salary (Polynominal Regression):', sc_y.inverse_transform(lin_reg_2.predict(poly_reg.fit_transform(sc_X.transform([[Level]]))).reshape(-1, 1))[0][0])

# Visualising the Linear Regression results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.scatter([Level],sc_y.inverse_transform(lin_reg.predict(sc_X.transform([[Level]]))), color='green', s=200, marker='*') # this has been added to show the single prediction result
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(lin_reg.predict(X).reshape(-1, 1)), color = 'blue')
plt.title('Estimated Salary (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.scatter([Level], sc_y.inverse_transform(lin_reg_2.predict(poly_reg.fit_transform(sc_X.transform([[Level]]))).reshape(-1, 1)), color='green', s=200, marker='*') # this has been added to show the single prediction result
plt.plot(X_grid, sc_y.inverse_transform(lin_reg_2.predict(poly_reg.fit_transform(sc_X.transform(X_grid)))), color = 'blue') 
plt.title('Estimated Salary (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()