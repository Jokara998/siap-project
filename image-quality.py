import imquality.brisque as brisque
import PIL.Image
from utils import get_json_data
import numpy as np
from os.path import exists
import json

posts = get_json_data('./datasets/dataset_all.json')


def normalize(value, data):
    return (value - np.min(data)) / (np.max(data) - np.min(data))


def calculate_quality():
    data = []
    for post in posts:
        path = './images/' + post['shortCode'] + ".jpeg"
        quality = 0.50
        if exists(path):
            img = PIL.Image.open(path)
            quality = brisque.score(img)
            quality = round(quality, 2)
        data.append(quality)
    return data


def normalize_data(data):
    dataset = []
    for index, post in enumerate(posts):
        quality = normalize(data[index], data)
        quality = round(quality, 2)
        dataset.append({
            "quality": quality,
            "shortCode": post['shortCode']
        })
    return dataset


data = calculate_quality()
data = normalize_data(data)

with open('./datasets/dataset_images_quality.json', "w", encoding='utf8') as file:
    json.dump(data, file, indent=4)
