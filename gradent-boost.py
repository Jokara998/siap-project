import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from dataset import get_dataset, get_features

X_train, X_test, y_train, y_test = get_dataset()
features = get_features()
rfr = GradientBoostingRegressor(max_depth=3,
                                alpha=0.7,
                                learning_rate=0.21, loss='huber', n_estimators=100, random_state=1).fit(X_train, y_train)
score = rfr.score(X_train, y_train)
print("R-squared:", score)


importances = list(rfr.feature_importances_)
[print(f'{f}: {i}')
 for f, i in zip(features, importances)]


predictions = np.floor(rfr.predict(X_test))

print('Mean Absolute Error (MAE):',
      metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(
    metrics.mean_squared_error(y_test, predictions)))
mape = np.mean(np.abs((y_test - predictions) / np.abs(y_test)))
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))
