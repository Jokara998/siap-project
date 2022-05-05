import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from dataset import get_dataset, get_features

X_train, X_test, y_train, y_test = get_dataset()
features = get_features()
rfr = RandomForestRegressor(max_features=10, max_samples=0.75,
                            n_estimators=320,  ccp_alpha=1.14, max_depth=11, random_state=1).fit(X_train, y_train)
score = rfr.score(X_train, y_train)

# uncomment for hyperparam optimization
# param_grid = {
#     'ccp_alpha': [
#         1, 1.14, 1.2, 1.5],
#     'max_features': [3, 7, 9, 10, 11],
#     'max_depth': [7, 9, 11, 15],
#     'n_estimators': [10, 100, 320, 350],
#     'max_samples': [0.1, 0.3, 0.5, 0.75, 0.9, 1]
# }

# grid_clf = GridSearchCV(RandomForestRegressor(), param_grid, cv=10, verbose=2)
# grid_clf.fit(X_train, y_train)

# print(grid_clf. best_params_)

importances = list(rfr.feature_importances_)
[print(f'{f}: {i}')
 for f, i in zip(features, importances)]


predictions = np.floor(rfr.predict(X_test))


features = ['hashtags', 'mentions', 'comment count', 'followers',
            'following', 'post count', 'weekday', 'avg_comments', 'quality', 'object', 'y']


# best/worst results
abs_vals = abs(predictions-y_test)
pair = sorted(zip(abs_vals, X_test, y_test), key=lambda x: x[0])
xx = []
for i in range(10):
    xd = pair[-i-1]  # for worst
    #xd = pair[i]
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
