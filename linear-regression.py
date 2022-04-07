import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from utils import get_json_data

posts = get_json_data('./datasets/dataset_all.json')
X = []
Y = []
for post in posts:
    x = [len(post['hashtags']), len(post['mentions']), post['commentsCount'],
         post['profile']['followersCount'], post['profile']['followsCount'], post['profile']['postsCount'], post['weekday'], post['avg_comment']]
    y = post['likesCount']
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=62)

reg = LinearRegression().fit(X_train, y_train)

print('Mean squared error: ')
print(mean_squared_error(y_test, np.floor(reg.predict(X_test))))
print('Root Mean Squared Error: ')

print(mean_squared_error(y_test, np.floor(reg.predict(X_test)), squared=False))
print('Mean absoulte error: ')
print(mean_absolute_error(y_test, np.floor(reg.predict(X_test))))
