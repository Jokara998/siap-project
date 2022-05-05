import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from dataset import get_dataset, get_features

features2 = ['hashtags', 'mentions', 'comment count', 'followers',
             'following', 'post count', 'weekday', 'avg_comments', 'quality', 'object', 'y']


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


# best/worst results
abs_vals = abs(predictions-y_test)
pair = sorted(zip(abs_vals, X_test, y_test), key=lambda x: x[0])
xx = []
for i in range(10):
    # xd=pair[-i-1] for worst
    xd = pair[-1-i]
    a = list(xd[1])
    a.append(xd[2])
    xx.append(a)

df = pd.DataFrame(xx)
df.columns = features2
print(df.head(10))


print('Mean Absolute Error (MAE):',
      metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(
    metrics.mean_squared_error(y_test, predictions)))
mape = np.mean(np.abs((y_test - predictions) / np.abs(y_test)))
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))
