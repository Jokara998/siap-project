import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import get_dataset, get_features

X_train, X_test, y_train, y_test = get_dataset()
features = get_features()
rfr = RandomForestRegressor(max_features=3,
                            n_estimators=123, ccp_alpha=0.2, random_state=1).fit(X_train, y_train)
score = rfr.score(X_train, y_train)
print("R-squared:", score)


importances = list(rfr.feature_importances_)
[print(f'{f}: {i}')
 for f, i in zip(features, importances)]


print('Mean squared error: ')
print(mean_squared_error(y_test, np.floor(rfr.predict(X_test))))
print('Root Mean Squared Error: ')

print(mean_squared_error(y_test, np.floor(rfr.predict(X_test)), squared=False))
print('Mean absoulte error: ')
print(mean_absolute_error(y_test, np.floor(rfr.predict(X_test))))
