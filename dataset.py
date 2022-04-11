import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import get_json_data

features = ['hastags', 'mentions', 'comment count', 'followers',
            'following', 'post count', 'weekday', 'avg_comments', 'quality', 'object']


def get_features():
    return features


def get_dataset():
    posts = get_json_data('./datasets/dataset_all.json')
    qualities = get_json_data('./datasets/dataset_images_quality.json')
    image_labels = get_json_data('./datasets/dataset_images_objects.json')
    image_labels_list = [mapping['object'] for mapping in image_labels]
    encoded = LabelEncoder().fit(image_labels_list)
    X = []
    Y = []

    for post in posts:
        q = [qual for qual in qualities if qual['shortCode'] == post['shortCode']]
        quality = q[0]['quality']

        lab = [lab for lab in image_labels if lab['shortCode'] == post['shortCode']]

        obj = encoded.transform([lab[0]['object']])

        x = [len(post['hashtags']), len(post['mentions']), post['commentsCount'],
             post['profile']['followersCount'], post['profile']['followsCount'], post['profile']['postsCount'], post['weekday'], post['avg_comment'], quality, obj[0]]
        y = post['likesCount']

        if(y > 1000 or y < 50):
            continue
        X.append(x)
        Y.append(y)

    minimum = min(Y)
    maximum = max(Y)
    r = range(minimum, maximum, 10)
    n, bins, patches = plt.hist(x=Y, bins=list(r), color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    # plt.show()

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=62)

    return [X_train, X_test, y_train, y_test]
