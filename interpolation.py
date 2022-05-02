import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from dataset import get_dataset

X_train, X_test, y_train, y_test = get_dataset()

model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3))
model.fit(X_train, y_train)

print('Mean squared error: ')
print(mean_squared_error(y_test, np.floor(model.predict(X_test))))
print('Root Mean Squared Error: ')

print(mean_squared_error(y_test, np.floor(model.predict(X_test)), squared=False))
print('Mean absoulte error: ')
print(mean_absolute_error(y_test, np.floor(model.predict(X_test))))
