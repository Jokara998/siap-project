import numpy as np
from sklearn.model_selection import train_test_split

from utils import get_json_data

features = ['hastags', 'mentions', 'comment count', 'followers',
            'following', 'post count', 'weekday', 'avg_comments']


def get_features():
    return features


def get_dataset():
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

    return [X_train, X_test, y_train, y_test]
