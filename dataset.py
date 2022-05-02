import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import get_json_data

features = ['hashtags', 'mentions', 'comment count', 'followers',
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
        if(post['profile']['postsCount'] > 200):
            continue
        if(y > 800 or y < 50):
            continue
        if(quality > 0.85):
            continue
        X.append(x)
        Y.append(y)

    minimum = min(Y)
    maximum = max(Y)

    # r = range(minimum, maximum, 10)
    # df = pd.DataFrame(X)
    # df.columns = features
   # print(df.head())
  #  n, bins, patches = plt.hist(x=Y, color='#0504aa',
   #                             alpha=0.7, rwidth=0.85)
   # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Value')
  #  ##plt.ylabel('Frequency')
    # plt.title('My Very Own Histogram')
   # plt.show()
    # print(len(Y))
    # print(statistics.median(df['followers']))

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=62)

    return [X_train, X_test, y_train, y_test]


def showData():
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
        if(post['profile']['postsCount'] > 200):
            continue
        if(y > 800 or y < 50):
            continue
        if(quality > 0.85):
            continue
        X.append(x)
        Y.append(y)

    minimum = min(Y)
    maximum = max(Y)

    r = range(minimum, maximum, 10)
    df = pd.DataFrame(X)
    df.columns = features
    df['y'] = Y

    sn.heatmap(df.corr(), annot=True)
    plt.show()

    print('Mean: ', statistics.mean(Y))
    print('Median: ', statistics.median(Y))

    # ----------------------------
    # lajk / post
    plt.hist(x=Y, color='#0504aa', bins=list(r))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Broj lajkova')
    plt.ylabel('Broj postova')
    plt.title('Lajk / Post')
    plt.show()
    # -----------------------------

    # komentar / post
    plt.hist(x=df['comment count'], color='#0504aa',
             bins=list(range(0, 100)), rwidth=0.95)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Broj Komentara')
    plt.ylabel('Broj postova')
    plt.title('Komentari / Post')
    plt.show()
    # --------------------

    # HASTTAG / post
    plt.hist(x=df['hashtags'], color='#0504aa',
             bins=list(range(0, 35)), rwidth=0.9)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Broj Hashtagova')
    plt.ylabel('Broj postova')
    plt.title('Hashtag / Post')
    plt.show()
    # --------------------

    # followers / post
    plt.hist(x=df['quality'], color='#0504aa',
             rwidth=0.9)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Kvalitet')
    plt.ylabel('Broj postova')
    plt.title('Kvalitet / Post')
    plt.show()
    # --------------------

    # post count / post
    plt.hist(x=df['post count'], color='#0504aa',
             rwidth=0.9)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Broj postova vlasnika')
    plt.ylabel('Postovi')
    plt.title('Broj postova ')
    plt.show()
    # --------------------

    # plt.bar(x=df['followers'], y=Y)
    # print(type(df['followers']))
    # print(len(df['followers']))
    print(len(Y))
    data = list(zip(list(df['followers']), Y))
    data = sorted(data, key=lambda x:  x[1])

    xxx = [x[0] for x in data]
    yyy = [x[1] for x in data]

    plt.bar(list(range(0, len(xxx))), xxx,
            align='edge', width=0.4, color='g')
    plt.bar(list(range(0, len(xxx))), yyy,
            align='edge', width=0.4, color='c')
    # plt.bar([0, 1, 2, 3], df.x4, align='edge', width=-0.4, color='r')
    # plt.bar([0, 1, 2, 3], df.x3, align='edge', width=-0.4, color='y')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('lajkovi')
    plt.ylabel('pratioci')
    plt.title('Pratioci / lajkovi')
    plt.show()

    # print(len(Y))
    # print(statistics.median(df['followers']))
if __name__ == '__main__':
    showData()
