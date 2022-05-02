import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from dataset import get_dataset, get_features

X_train, X_test, y_train, y_test = get_dataset()
features = get_features()
rfr = SVR(gamma='auto', kernel='rbf', degree=3,
          C=100, epsilon=0.5).fit(X_train, y_train)
score = rfr.score(X_train, y_train)
print("R-squared:", score)


predictions = np.floor(rfr.predict(X_test))

print('Mean Absolute Error (MAE):',
      metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(
    metrics.mean_squared_error(y_test, predictions)))
mape = np.mean(np.abs((y_test - predictions) / np.abs(y_test)))
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))
