import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression

from dataset import get_dataset

X_train, X_test, y_train, y_test = get_dataset()

reg = LinearRegression().fit(X_train, y_train)

predictions = np.floor(reg.predict(X_test))

print('Mean Absolute Error (MAE):',
      metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(
    metrics.mean_squared_error(y_test, predictions)))
mape = np.mean(np.abs((y_test - predictions) / np.abs(y_test)))
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))
