from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from utils import get_json_data
from os.path import exists
import json

images = get_json_data('./datasets/dataset_images_quality.json')
model = VGG16()
data = []
for img in images:
    path = './images/'+img['shortCode']+'.jpeg'
    obj = 'seashore'

    if exists(path):
        image = load_img(path, target_size=(224, 224, 3))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        prediction = model.predict(image)
        label = decode_predictions(prediction)
        label = label[0][0]
        obj = label[1]

    data.append({
        "object": obj,
        "shortCode": img['shortCode']
    })

with open('./datasets/dataset_images_objects.json', "w", encoding='utf8') as file:
    json.dump(data, file, indent=4)