from re import X

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression

from dataset import get_dataset

features = ['hashtags', 'mentions', 'comment count', 'followers',
            'following', 'post count', 'weekday', 'avg_comments', 'quality', 'object', 'y']


X_train, X_test, y_train, y_test = get_dataset()

reg = LinearRegression().fit(X_train, y_train)

predictions = np.floor(reg.predict(X_test))

# best/worst results
abs_vals = abs(predictions-y_test)
pair = sorted(zip(abs_vals, X_test, y_test), key=lambda x: x[0])
xx = []
for i in range(10):
    # xd=pair[-i-1] for worst
    xd = pair[i]
    a = list(xd[1])
    a.append(xd[2])
    xx.append(a)

df = pd.DataFrame(xx)
df.columns = features
print(df.head(10))

print('Mean Absolute Error (MAE):',
      metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(
    metrics.mean_squared_error(y_test, predictions)))
mape = np.mean(np.abs((y_test - predictions) / np.abs(y_test)))
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))
