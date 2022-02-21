import json


def get_json_data(path):
    with open(path) as file:
        return json.load(file)
