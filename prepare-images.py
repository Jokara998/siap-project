from bs4 import BeautifulSoup
import requests
from selenium import webdriver
import time
from utils import get_json_data
from os.path import exists

posts = get_json_data('./datasets/dataset_all.json')
config = get_json_data('config.json')

driver = webdriver.Chrome(executable_path='./chromedriver')
driver.maximize_window()
driver.get('https://www.instagram.com/')
time.sleep(5)

username_input = driver.find_element_by_css_selector("input[name='username']")
password_input = driver.find_element_by_css_selector("input[name='password']")

username_input.send_keys(config['instagram']['username'])
password_input.send_keys(config['instagram']['password'])

login_button = driver.find_element_by_xpath("//button[@type='submit']")
login_button.click()
time.sleep(10)

for post in posts:
    file_exists = exists("./images/" + post['shortCode'] + ".png")
    if file_exists:
        continue

    driver.get(post['url'])
    soup = BeautifulSoup(driver.page_source, 'lxml')
    img = soup.find('img', class_='FFVAD')
    if img is not None:
        img_url = img['src']
        r = requests.get(img_url)
        image = r.content
        with open("./images/" + post['shortCode'] + ".jpeg", 'wb') as f:
            f.write(image)