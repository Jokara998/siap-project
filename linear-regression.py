import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import get_dataset

X_train, X_test, y_train, y_test = get_dataset()

reg = LinearRegression().fit(X_train, y_train)

print('Mean squared error: ')
print(mean_squared_error(y_test, np.floor(reg.predict(X_test))))
print('Root Mean Squared Error: ')

print(mean_squared_error(y_test, np.floor(reg.predict(X_test)), squared=False))
print('Mean absoulte error: ')
print(mean_absolute_error(y_test, np.floor(reg.predict(X_test))))
