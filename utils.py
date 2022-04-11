import json

import numpy as np


def get_json_data(path):
    with open(path) as file:
        return json.load(file)


def normalize(value, data):
    return (value - np.min(data)) / (np.max(data) - np.min(data))
