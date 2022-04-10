import imquality.brisque as brisque
import PIL.Image
from utils import get_json_data
posts = get_json_data('./datasets/dataset_all.json')

for post in posts:
    path = './images/'+post['shortCode']+".jpeg"
    img = PIL.Image.open(path)
    if img is not None:
        print(brisque.score(img))