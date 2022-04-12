from utils import get_json_data
import json

profiles = get_json_data('./new_profiles.json')

for profile in profiles:
    posts = []
    for post in profile['latestPosts']:
        if post['type'] == 'Image':
            posts.append(post)

with open('./new_posts.json', "w", encoding='utf8') as file:
    json.dump(posts, file, indent=4)

comments = []
for post in posts:
    comments.append('https://www.instagram.com/p/'+post['shortCode'])

with open('./new_comments.json', "w", encoding='utf8') as file:
    json.dump(comments, file, indent=4)